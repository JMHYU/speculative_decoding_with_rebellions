import traceback
import warnings
from typing import List, Optional, Union

import torch
from transformers.generation import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import SampleDecoderOnlyOutput


class RBLNGenerationMixin:
    # call 'greedy_search` directly is deprecated and removed in v4.41.
    def greedy_search(self, *args, **kwargs):
        return self._greedy_search(*args, **kwargs)

    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        generation_config: Optional[GenerationConfig] = None,  # thkim change for 4.41.0
        **model_kwargs,
    ) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:
        ###################### thkim change for 4.41.0 ############################
        if generation_config is not None:
            pad_token_id = generation_config.pad_token_id
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
        ##########################################################################
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only

        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            try:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
            except Exception:
                traceback.print_exc()
                break

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (next_token_logits,)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            ########################################################################################################
            # thkim change for right-padding batch
            # if min_input_len <= update_idx < max_input_len
            #   update validate input_ids[:,update_idx]
            # TODO : raw_logits contains dummy next_token's logits
            if hasattr(self, "rightpad_max_len"):
                update_idx = model_inputs["cache_position"] + model_inputs["query_length"]
                if update_idx < self.rightpad_max_len:
                    # update exist input_ids rather than concat
                    valid_indices = model_kwargs["attention_mask"][:, update_idx] == 0
                    dummy_indices = model_kwargs["attention_mask"][:, update_idx] == 1

                    input_ids[valid_indices, update_idx] = next_tokens[valid_indices]
                    model_kwargs["attention_mask"][valid_indices, update_idx] = 1
                    model_kwargs["past_key_values"] = outputs["past_key_values"]

                    # dummy next_token -> pad_token_id for streamer
                    # in order to skip by 'skip_special_tokens = True"
                    if streamer is not None:
                        next_tokens[dummy_indices] = pad_token_id
                else:
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs,
                        model_kwargs,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                    )
            else:
                ############################################END#########################################################
                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

            if streamer is not None:
                streamer.put(next_tokens.cpu())
                if streamer.is_blocked():
                    this_peer_finished = True

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                ####################################################################
                # thkim : to do not finish sequence of dummy_decoder of right_padding
                if hasattr(self, "rightpad_max_len"):
                    update_idx = model_inputs["cache_position"] + model_inputs["query_length"]
                    if update_idx < self.rightpad_max_len:
                        next_tokens += (
                            model_kwargs["attention_mask"][:, update_idx] * self.generation_config.eos_token_id
                        )
                ######################################################################
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            # thkim : backward compatibility bool vs torch.BoolTensor
            is_stop = stopping_criteria(input_ids, None)
            if isinstance(is_stop, torch.BoolTensor):
                is_stop = torch.all(is_stop)
            if is_stop:
                this_peer_finished = True

            if this_peer_finished:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            ############## thkim : roate raw_logits when right_padding#####################
            if hasattr(self, "rightpad_max_len"):
                raw_logits = torch.stack(raw_logits).transpose(0, 1)
                for i in range(input_ids.shape[0]):
                    raw_logits[i] = torch.cat((raw_logits[i][self.dummy_len[i] :], raw_logits[i][: self.dummy_len[i]]))
                raw_logits = raw_logits.transpose(1, 0)
            ##################################################################################
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                logits=raw_logits,
            )
        else:
            return input_ids

    # call 'sample` directly is deprecated and removed in v4.41.
    def sample(self, *args, **kwargs):
        return self._sample(*args, **kwargs)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        generation_config: Optional[GenerationConfig] = None,
        do_sample: Optional[bool] = True,
        **model_kwargs,
    ) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:
        ###################### thkim change for 4.41.0 ############################
        if generation_config is not None:
            pad_token_id = generation_config.pad_token_id
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            do_sample = generation_config.do_sample
        ###########################################################################

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else False

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        this_peer_finished = False

        # model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            try:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits = outputs.logits[:, -1, :]
            except Exception:
                traceback.print_exc()
                break

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            ###################### thkim change for 4.41.0 ############################
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)
            ###########################################################################

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)

            # sample
            ###################### thkim change for 4.41.0 ############################
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            ###########################################################################

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            ###############################thkim change for right-padding batch#################################
            # if min_input_len <= update_idx < max_input_len
            #   update validate input_ids[:,update_idx]
            # TODO : raw_logits contains dummy next_token's logits

            if hasattr(self, "rightpad_max_len"):
                update_idx = model_inputs["cache_position"] + model_inputs["query_length"]
                if update_idx < self.rightpad_max_len:
                    # update exist input_ids rather than concat
                    valid_indices = model_kwargs["attention_mask"][:, update_idx] == 0
                    dummy_indices = model_kwargs["attention_mask"][:, update_idx] == 1

                    input_ids[valid_indices, update_idx] = next_tokens[valid_indices]
                    model_kwargs["attention_mask"][valid_indices, update_idx] = 1
                    model_kwargs["past_key_values"] = outputs["past_key_values"]
                    # dummy next_token -> pad_token_id for streamer
                    # in order to skip by 'skip_special_tokens = True"
                    if streamer is not None:
                        next_tokens[dummy_indices] = pad_token_id
                else:
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs,
                        model_kwargs,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                    )
            else:
                ############################################END#########################################################
                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

            if streamer is not None:
                streamer.put(next_tokens.cpu())
                if streamer.is_blocked():
                    this_peer_finished = True

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                ####################################################################
                # thkim : to do not finish sequence of dummy_decoder of right_padding
                if hasattr(self, "rightpad_max_len"):
                    update_idx = model_inputs["cache_position"] + model_inputs["query_length"]
                    if update_idx < self.rightpad_max_len:
                        next_tokens += (
                            model_kwargs["attention_mask"][:, update_idx] * self.generation_config.eos_token_id
                        )

                ######################################################################
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            # thkim : backward compatibility bool vs list[bool]
            is_stop = stopping_criteria(input_ids, None)
            if isinstance(is_stop, torch.BoolTensor):
                is_stop = torch.all(is_stop)
            if is_stop:
                this_peer_finished = True

            if this_peer_finished:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            ############## thkim : roate raw_logits when right_padding#####################
            if hasattr(self, "rightpad_max_len"):
                raw_logits = torch.stack(raw_logits).transpose(0, 1)
                for i in range(input_ids.shape[0]):
                    raw_logits[i] = torch.cat((raw_logits[i][self.dummy_len[i] :], raw_logits[i][: self.dummy_len[i]]))
                raw_logits = raw_logits.transpose(1, 0)
            ##################################################################################
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
            )
        else:
            return input_ids
