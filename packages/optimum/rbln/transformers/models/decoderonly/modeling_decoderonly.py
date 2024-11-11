# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.
import glob
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import rebel  # noqa: F401
import torch  # noqa: F401
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils import ModelOutput

from ....modeling_base import RBLNModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ....utils.runtime_utils import RBLNPytorchRuntime
from ....utils.timer_utils import rbln_timer


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )

SUPPORTED_QUANTIZATIONS = {
    "rbln": [
        "w4a16",
    ],
}


class RBLNRuntimeModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name", "embed_tokens"]

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        batch_position: torch.Tensor,
        query_idx: torch.Tensor,
        **kwargs,
    ):
        if inputs_embeds is None:
            inp = input_ids
            if self.embed_tokens is not None:
                inp = self.embed_tokens(inp)

            return super().forward(
                inp,
                attention_mask,
                cache_position,
                batch_position,
                query_idx,
                **kwargs,
            )
        else:
            return super().forward(
                inputs_embeds,
                attention_mask,
                cache_position,
                batch_position,
                query_idx,
                **kwargs,
            )


@dataclass
class RBLNDecoderOnlyOutput(ModelOutput):
    logits: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    past_cached_length: Union[int, torch.Tensor] = None


class RBLNDecoderOnlyModelForCausalLM(RBLNModel, ABC):
    """
    The DecoderOnly Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNMultiModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based DecoderOnlyForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers DecoderOnlyForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    main_input_name = "input_ids"
    auto_model_class = AutoModelForCausalLM

    def __post_init__(self, **kwargs):
        self.batch_size = self.rbln_config.model_cfg["batch_size"]
        self.max_seq_len = self.rbln_config.model_cfg["max_seq_len"]
        self.prefill_chunk_size = self.rbln_config.model_cfg["prefill_chunk_size"]

        self.prefill_attention_mask = torch.zeros(1, 1, self.prefill_chunk_size, self.max_seq_len, dtype=torch.float32)
        self.causal_mask = 1 - torch.triu(
            torch.ones(1, 1, self.prefill_chunk_size, self.prefill_chunk_size), diagonal=1
        )
        self.dec_attn_mask_init = torch.zeros(1, 1, 1, self.max_seq_len, dtype=torch.float32)
        self.dec_attn_mask = torch.zeros(self.batch_size, 1, 1, self.max_seq_len, dtype=torch.float32)

        main_input_name = self.main_input_name
        if self.rbln_config.model_cfg["use_inputs_embeds"]:
            main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            with no_init_weights():
                self.embed_tokens = torch.nn.Embedding(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    self.config.pad_token_id,
                )
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        self.prefill_decoder = RBLNRuntimeModel(
            runtime=self.model[0], main_input_name=main_input_name, embed_tokens=self.embed_tokens
        )
        self.decoder = RBLNRuntimeModel(
            runtime=self.model[1], main_input_name=main_input_name, embed_tokens=self.embed_tokens
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        if rbln_config.model_cfg["use_inputs_embeds"]:
            save_dict = {}
            save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
            torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def get_input_embeddings(self):
        return self.embed_tokens

    @classmethod
    def get_quantized_model(
        cls,
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        from ...utils.rbln_quantization import update_layers_to_quantized

        kwargs = cls.update_kwargs(kwargs)

        config = AutoConfig.from_pretrained(
            model_id,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        with no_init_weights():
            model = AutoModelForCausalLM.from_config(config)

        update_layers_to_quantized(model)

        n_layer = kwargs.get("num_hidden_layers", None)
        cls._load_weights_directly_to_model(model, model_id, n_layer)

        return model

    def _load_weights_directly_to_model(model, model_id, n_layer=None):
        """
        Load safetensor file data directly into the model, filtering by layer if n_layer is provided.
        """

        model_params = dict(model.named_parameters(recurse=True))
        model_buffers = dict(model.named_buffers(recurse=True))
        safetensor_files = glob.glob(f"{model_id}/*.safetensors")

        target_layers = list(range(n_layer)) if n_layer is not None else None

        for safetensor_file in safetensor_files:
            file_data = load_file(safetensor_file)
            for key, value in file_data.items():
                if target_layers is not None:
                    parts = key.split(".")

                    if len(parts) > 2 and parts[2].isdigit() and (int(parts[2]) not in target_layers):
                        continue

                if key in model_params:
                    model_params[key].data.copy_(value)
                elif key in model_buffers:
                    model_buffers[key].data.copy_(value)

        return 0

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs) -> "PreTrainedModel":
        rbln_kwargs = kwargs.get("rbln_kwargs", {})
        rbln_quantization = rbln_kwargs.get("quantization", None)

        if rbln_quantization is not None and rbln_quantization["format"] == "rbln":
            model = cls.get_quantized_model(*args, **kwargs)
        else:
            model = super().get_pytorch_model(*args, **kwargs)

        return model

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_rbln_compile_config = rbln_compile_configs[0]
        dec_rbln_compile_config = rbln_compile_configs[1]

        @rbln_timer("Jit Trace")
        def get_scripted_model():
            # This function is nested to dealloc the example inputs before compilation.
            # [AIHA-yjh] dummy inputs 생성. 41개의 인풋을 생성한다.
            # [AIHA-yjh] 그 이유는 input_ids_or_inputs_embeds, attention_mask, cache_position, batch_position, query_idx 5개 + past_key_values 36개
            prefill_example_inputs = prefill_rbln_compile_config.get_dummy_inputs(fill=0)
            dec_example_inputs = dec_rbln_compile_config.get_dummy_inputs(fill=4)


            batch_index = 3
            dec_example_inputs[batch_index].fill_(-1)  # fill batch_position -1 to indicate it is decoder.

            prefill_scripted_model = torch.jit.trace(
                wrapped_model, prefill_example_inputs, check_trace=False, _store_inputs=False
            )
            dec_scripted_model = torch.jit.trace(
                wrapped_model, dec_example_inputs, check_trace=False, _store_inputs=False
            )
            return prefill_scripted_model, dec_scripted_model

        prefill_scripted_model, dec_scripted_model = get_scripted_model()

        @rbln_timer("TorchScript to IR")
        def scripted_model_to_ir():
            prefill_ir = rebel.torchscript_to_ir(
                prefill_scripted_model,
                input_names=[v[0] for v in prefill_rbln_compile_config.input_info],
            )
            dec_ir = rebel.torchscript_to_ir(
                dec_scripted_model,
                input_names=[v[0] for v in dec_rbln_compile_config.input_info],
            )
            return prefill_ir, dec_ir

        prefill_ir, dec_ir = scripted_model_to_ir()
        # Caching prefill_decoder/decoder I/O
        cache_index_offset = 5
        connections = [
            (prefill_ir.outputs[1 + i], prefill_ir.inputs[cache_index_offset + i])
            for i in range(model.config.num_hidden_layers * 2)
        ]

        compiled_model = rebel.compile(
            prefill_ir,
            dec_ir,
            connections=connections,
            fusion=prefill_rbln_compile_config.fusion,
            npu=prefill_rbln_compile_config.npu,
            tensor_parallel_size=prefill_rbln_compile_config.tensor_parallel_size,
            use_weight_sharing=True,
        )
        return compiled_model

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_quantization = rbln_kwargs.get("quantization", None)
        rbln_use_inputs_embeds = rbln_kwargs.get("use_inputs_embeds", None)

        prefill_chunk_size = 128
        if rbln_max_seq_len is None:
            rbln_max_seq_len = getattr(model_config, "max_position_embeddings", None) or getattr(
                model_config, "n_positions", None
            )
        if rbln_max_seq_len is None:
            raise ValueError("`rbln_max_seq_len` should be specified.")
        rbln_batch_size = 1 if rbln_batch_size is None else rbln_batch_size
        rbln_use_inputs_embeds = False if rbln_use_inputs_embeds is None else rbln_use_inputs_embeds

        num_attention_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads")
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers")
        head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // num_attention_heads
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size")

        if rbln_quantization is not None:
            q_format = rbln_quantization.get("format", None)
            q_precision = rbln_quantization.get("precision", None)

            if q_format not in SUPPORTED_QUANTIZATIONS.keys() or q_precision not in SUPPORTED_QUANTIZATIONS[q_format]:
                raise ValueError(
                    f'rbln_quantization="{rbln_quantization}" is not a supported quantization format or precesion, '
                    f"Possible: {SUPPORTED_QUANTIZATIONS}"
                )

        def get_input_info(
            batch_size,
            query_length,
            use_inputs_embeds,
            hidden_size,
        ):
            if use_inputs_embeds:
                main_input = ("inputs_embeds", [batch_size, query_length, hidden_size], "float32")
            else:
                main_input = ("input_ids", [batch_size, query_length], "int64")

            input_info = [
                main_input,
                ("attention_mask", [batch_size, 1, query_length, rbln_max_seq_len], "float32"),
                (
                    "cache_position",
                    [batch_size, query_length],
                    "int32",
                ),
                ("batch_position", [], "int16"),
                ("query_idx", [], "int16"),
            ]

            input_info.extend(
                [
                    (
                        f"past_key_values_{i}",
                        [
                            rbln_batch_size,
                            num_key_value_heads,
                            rbln_max_seq_len,
                            head_dim,
                        ],
                        "float32",
                    )
                    for i in range(num_hidden_layers * 2)
                ]
            )

            return input_info

        prefill_input_info = get_input_info(
            batch_size=1,
            query_length=prefill_chunk_size,
            use_inputs_embeds=rbln_use_inputs_embeds,
            hidden_size=hidden_size,
        )
        dec_input_info = get_input_info(
            batch_size=rbln_batch_size,
            query_length=1,
            use_inputs_embeds=rbln_use_inputs_embeds,
            hidden_size=hidden_size,
        )

        prefill_rbln_compile_config = RBLNCompileConfig(input_info=prefill_input_info)
        dec_rbln_compile_config = RBLNCompileConfig(input_info=dec_input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[prefill_rbln_compile_config, dec_rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update(
            {
                "max_seq_len": rbln_max_seq_len,
                "batch_size": rbln_batch_size,
                "prefill_chunk_size": prefill_chunk_size,
                "use_inputs_embeds": rbln_use_inputs_embeds,
            }
        )

        if rbln_quantization is not None:
            rbln_config.model_cfg.update({"quantization": rbln_quantization})

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls, compiled_models: List[rebel.RBLNCompiledModel], rbln_device_map: Dict[str, int]
    ) -> List[rebel.Runtime]:
        device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
        return [
            compiled_models[0].create_runtime(input_info_index=0, tensor_type="pt", device=device_val),
            compiled_models[0].create_runtime(input_info_index=1, tensor_type="pt", device=device_val),
        ]

    def get_decoder(self):
        return self.decoder

    def can_generate(self):
        return True

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_cached_length: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        # prefill phase
        if past_cached_length is None:
            # huggingface make dummy_input_ids if model_input_name is "input_embeds"
            # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L469
            if self.rbln_config.model_cfg["use_inputs_embeds"] and inputs_embeds is not None:
                input_tensors = inputs_embeds
            else:
                input_tensors = input_ids

            batch_size = input_tensors.shape[0]
            l_input_tensors = []
            cache_positions = []
            past_cached_length = torch.zeros((batch_size, 1), dtype=torch.int32)
            for i in range(batch_size):
                input_tensor = input_tensors[i]
                input_tensor = input_tensor[attention_mask[i] == 1]
                valid_len = input_tensor.shape[0]
                cache_position = torch.arange(0, valid_len, dtype=torch.int32)
                past_cached_length[i] = valid_len
                l_input_tensors.append(input_tensor.unsqueeze(0))
                cache_positions.append(cache_position.unsqueeze(0))

            input_tensors = l_input_tensors
            if self.rbln_config.model_cfg["use_inputs_embeds"] and inputs_embeds is not None:
                model_inputs.update({"inputs_embeds": input_tensors, "input_ids": input_ids})
            else:
                model_inputs.update({"input_ids": input_tensors, "inputs_embeds": inputs_embeds})
        # decoder phase
        else:
            input_ids = input_ids[:, -1:]
            cache_positions = past_cached_length
            past_cached_length = past_cached_length + 1
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "cache_position": cache_positions,
                "past_cached_length": past_cached_length,
            }
        )

        # [AIHA-yjh] batch size 및 input token 길이에 맞춰서 해당 형태로 만들어주었다.
        # [AIHA-yjh] decoder phase에서의 cache_position: [tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]], dtype=torch.int32)]
        # [AIHA-yjh] prefill phase에서의 cache_position: [tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]], dtype=torch.int32)]

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # update past_cached_length
        model_kwargs["past_cached_length"] = outputs.past_cached_length

        return model_kwargs

    def forward(
        self,
        input_ids: Optional[Union[List[torch.LongTensor], torch.LongTensor]] = None,
        inputs_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        cache_position: Union[List[torch.Tensor], torch.Tensor] = None,  # vllm keyword argument
        batch_idx: Optional[int] = None,
        past_cached_length: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        # prefll & hf generate
        # [AIHA-yjh] kwargs로 output_hidden_states 옵션이 넘어오는 것까지 확인함. 하지만 kwargs를 사용하지 않는다.
    
        if isinstance(cache_position, list):
            logits = []
            last_hidden_states = []
            input_tensors = input_ids if inputs_embeds is None else inputs_embeds
            for batch_idx, (input_tensor, cache_pos) in enumerate(zip(input_tensors, cache_position)):
                # [AIHA-yjh] cache_pos = cache_position[0]
                logit, last_hidden_state = self._forward_prefill(
                    input_ids=input_tensor if inputs_embeds is None else None,
                    inputs_embeds=input_tensor if inputs_embeds is not None else None,
                    cache_position=cache_pos,
                    batch_idx=batch_idx,
                    **kwargs
                )
                logits.append(logit)
                last_hidden_states.append(last_hidden_state)
            logits = torch.cat(logits, dim=0)
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
        # prefill & vllm step
        elif cache_position.shape[-1] > 1:
            logits, last_hidden_state = self._forward_prefill(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                batch_idx=batch_idx,
            )
        # common decoder
        else:
            logits, last_hidden_states = self._forward_decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

        return RBLNDecoderOnlyOutput(
            logits=logits,
            last_hidden_states=last_hidden_states,
            past_cached_length=past_cached_length,
        )

    def _forward_prefill(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        batch_idx: int = None,
        **kwargs
    ) -> torch.FloatTensor:
        if batch_idx is None or batch_idx >= self.batch_size:
            raise RuntimeError(
                f"Invalid batch_idx ({batch_idx}). It must be a non-null value less than the batch size ({self.batch_size})."
            )
        # [AHIA-yjh] 이 값이 sync_runtime.py의 out으로 들어가는 것으로 추정된다.
        out_buffers = [
            torch.empty(
                size=[
                    1,
                    1,
                    self.config.vocab_size,
                ],
                dtype=torch.float32,
                device="cpu",
            ),
            torch.empty(size=[], dtype=torch.int16, device="cpu"),
            torch.empty(size=[1, 1, self.config.hidden_size], dtype=torch.float32, device="cpu"),
        ]


        if self.rbln_config.model_cfg["use_inputs_embeds"] and inputs_embeds is not None:
            model_input_name = "inputs_embeds"
        else:
            model_input_name = "input_ids"

        input_tensors = input_ids if model_input_name == "input_ids" else inputs_embeds

        query_length = input_tensors.shape[1]
        attention_mask = self.prefill_attention_mask.clone()

        for step in range(0, query_length, self.prefill_chunk_size):
            if step + self.prefill_chunk_size > query_length:
                # input_tensors = torch.nn.functional.pad(input_tensors, (0, step + self.prefill_chunk_size - query_length))
                padding_needed = step + self.prefill_chunk_size - query_length
                if model_input_name == "input_ids":
                    input_tensors = torch.nn.functional.pad(input_tensors, (0, padding_needed))
                else:
                    input_tensors = torch.nn.functional.pad(input_tensors, (0, 0, 0, padding_needed))

                # [AIHA-yjh] self.prefill_chunk_size가 128로 설정되어있다. 그래서 cache_position은 최종적으로 torch.arange(0, 128, dtype=torch.int32).unsqueeze(0)과 동일한 형태가 된다.
                cache_position = torch.cat(
                    [
                        cache_position,
                        torch.arange(
                            query_length,
                            step + self.prefill_chunk_size,
                            dtype=torch.int32,
                        ).unsqueeze(0),
                    ],
                    dim=-1,
                )

            sliced_input_tensors = input_tensors[:, step : step + self.prefill_chunk_size]
            sliced_cache_positions = cache_position[:, step : step + self.prefill_chunk_size]

            if step >= self.prefill_chunk_size:
                attention_mask[:, :, :, step - self.prefill_chunk_size : step] = 1
            attention_mask[:, :, :, step : step + self.prefill_chunk_size] = self.causal_mask

            query_idx = query_length % self.prefill_chunk_size - 1

            logits, _, last_hidden_state = self.prefill_decoder(
                input_ids=sliced_input_tensors.contiguous() if model_input_name == "input_ids" else None,
                inputs_embeds=sliced_input_tensors.contiguous() if model_input_name == "inputs_embeds" else None,
                attention_mask=attention_mask.contiguous(),
                cache_position=sliced_cache_positions.contiguous(),
                batch_position=torch.tensor(batch_idx, dtype=torch.int16),
                query_idx=torch.tensor(query_idx, dtype=torch.int16),
                out=out_buffers,
                **kwargs,
            )

        self.dec_attn_mask[batch_idx] = self.dec_attn_mask_init.clone()
        self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        return logits, last_hidden_state

    def _forward_decoder(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        cache_position: torch.Tensor = None,
    ) -> torch.FloatTensor:
        if self.rbln_config.model_cfg["use_inputs_embeds"] and inputs_embeds is not None:
            model_input_name = "inputs_embeds"
        else:
            model_input_name = "input_ids"
        input_tensors = input_ids if model_input_name == "input_ids" else inputs_embeds

        batch_size = input_tensors.shape[0]

        for b_idx in range(batch_size):
            decoding_step = cache_position[b_idx].item()
            self.dec_attn_mask[b_idx, :, :, decoding_step] = 1


        logits, _, hidden_state = self.decoder(
            input_ids=input_tensors.contiguous() if model_input_name == "input_ids" else None,
            inputs_embeds=input_tensors.contiguous() if model_input_name == "inputs_embeds" else None,
            attention_mask=self.dec_attn_mask.contiguous(),
            cache_position=cache_position.contiguous(),
            batch_position=torch.tensor(0, dtype=torch.int16),
            query_idx=torch.tensor(0, dtype=torch.int16),
        )

        return logits, hidden_state
