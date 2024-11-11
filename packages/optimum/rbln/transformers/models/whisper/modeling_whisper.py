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

import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import rebel
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    PretrainedConfig,
    WhisperForConditionalGeneration,
    WhisperModel,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from ....modeling_base import RBLNModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ....utils.runtime_utils import RBLNPytorchRuntime
from .generation_whisper import RBLNWhisperGenerationMixin
from .whisper_architecture import (
    _WhisperDecoderWrapper,
    _WhisperEncoderWrapper,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        PretrainedConfig,
    )


class RBLNRuntimeEncoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def forward(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        # backward compatibility transformers==4.40.2
        # https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/pipelines/automatic_speech_recognition.py#L494
        input_features = kwargs.get("input_features", None)
        if input_features is None:
            input_features = args[0]

        _ = super().forward(input_features=input_features)
        # dummy output for generation
        return BaseModelOutput(last_hidden_state=torch.tensor([[-1.0]]))


class RBLNRuntimeDecoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def forward(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        outputs = super().forward(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return Seq2SeqLMOutput(logits=outputs, cross_attentions=None)
        else:
            return Seq2SeqLMOutput(logits=outputs[0], cross_attentions=outputs[1])


class RBLNWhisperForConditionalGeneration(RBLNModel, RBLNWhisperGenerationMixin):
    """
    The Whisper Model with a language modeling head. Can be used for automatic speech recognition.
    This model inherits from [`RBLNMultiModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based LlamaForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers LlamaForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    model_type = "rbln_model"
    auto_model_class = AutoModelForSpeechSeq2Seq
    main_input_name = "input_ids"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        self.batch_size = self.rbln_config.model_cfg["batch_size"]
        self.dec_max_seq_len = self.rbln_config.model_cfg["dec_max_seq_len"]
        self.rbln_token_timestamps = self.rbln_config.model_cfg["token_timestamps"]

        self.encoder = RBLNRuntimeEncoder(runtime=self.model[0], main_input_name="input_features")
        self.decoder = RBLNRuntimeDecoder(runtime=self.model[1], main_input_name="input_ids")

        # skip encoder &  first decoder when language detected
        self.is_language_detected = False
        self.language_cross = None

        # Used in GenerationMixin.generate()
        # transformers/models/whisper/generation_whisper.py, line 505, in generate
        #     input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        self.model = WhisperModel(self.config)
        self.pad_token_id = self.config.pad_token_id

    def can_generate(self):
        return True

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def __getattr__(self, __name: str) -> Any:
        """This is the key method to implement RBLN-Whisper.
        Returns:
            Any: Whisper's corresponding method
        """

        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(WhisperForConditionalGeneration, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def _reorder_cache(self, past_key_values, beam_idx):
        # TODO(jongho): implement
        raise NotImplementedError

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        rbln_token_timestamps = rbln_config.model_cfg["token_timestamps"]
        wrapped_encoder = _WhisperEncoderWrapper(model).eval()
        wrapped_decoder = _WhisperDecoderWrapper(model, output_attentions=rbln_token_timestamps).eval()

        enc_rbln_compile_config = rbln_config.compile_cfgs[0]
        dec_rbln_compile_config = rbln_config.compile_cfgs[1]

        enc_example_inputs = enc_rbln_compile_config.get_dummy_inputs(fill=1)
        dec_example_inputs = dec_rbln_compile_config.get_dummy_inputs(fill=1)

        enc_scripted_model = torch.jit.trace(wrapped_encoder, enc_example_inputs, check_trace=False)
        dec_scripted_model = torch.jit.trace(wrapped_decoder, dec_example_inputs, check_trace=False)

        enc_ir = rebel.torchscript_to_ir(
            enc_scripted_model,
            input_names=[v[0] for v in enc_rbln_compile_config.input_info],
            name=enc_rbln_compile_config.mod_name,
        )
        dec_ir = rebel.torchscript_to_ir(
            dec_scripted_model,
            input_names=[v[0] for v in dec_rbln_compile_config.input_info],
            name=dec_rbln_compile_config.mod_name,
        )

        # Caching encoder/decoder I/O
        connections = [
            (enc_ir.outputs[0], dec_ir.inputs[4]),
            (dec_ir.outputs[1], dec_ir.inputs[3]),
        ]
        compiled_model = rebel.compile(
            enc_ir,
            dec_ir,
            connections=connections,
            fusion=enc_rbln_compile_config.fusion,
            npu=enc_rbln_compile_config.npu,
            tensor_parallel_size=enc_rbln_compile_config.tensor_parallel_size,
        )
        return compiled_model

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_token_timestamps = rbln_kwargs.get("token_timestamps", False)
        rbln_batch_size = 1 if rbln_batch_size is None else rbln_batch_size

        expected_seq_len = model_config.max_source_positions * 2
        num_mel_bins = model_config.num_mel_bins
        enc_max_seq_len = model_config.max_source_positions
        rbln_dec_max_seq_len = model_config.max_length

        # model input info
        enc_input_info = [("input_features", [rbln_batch_size, num_mel_bins, expected_seq_len], "float32")]
        dec_input_info = [
            ("decoder_input_ids", [rbln_batch_size, 1], "int64"),
            ("decoder_attention_mask", [rbln_batch_size, rbln_dec_max_seq_len], "int64"),
            ("cache_position", [], "int32"),
        ]
        dec_input_info.extend(
            [
                (
                    "self_key_value_states",
                    [
                        model_config.decoder_layers * 2,
                        rbln_batch_size,
                        model_config.decoder_attention_heads,
                        rbln_dec_max_seq_len,
                        model_config.d_model // model_config.encoder_attention_heads,
                    ],
                    "float32",
                )
            ]
        )
        dec_input_info.extend(
            [
                (
                    "cross_key_value_states",
                    [
                        model_config.decoder_layers * 2,
                        rbln_batch_size,
                        model_config.decoder_attention_heads,
                        enc_max_seq_len,
                        model_config.d_model // model_config.encoder_attention_heads,
                    ],
                    "float32",
                )
            ]
        )

        enc_rbln_compile_config = RBLNCompileConfig(mod_name="encoder", input_info=enc_input_info)
        dec_rbln_compile_config = RBLNCompileConfig(mod_name="decoder", input_info=dec_input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[enc_rbln_compile_config, dec_rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update(
            {
                "batch_size": rbln_batch_size,
                "dec_max_seq_len": rbln_dec_max_seq_len,
                "token_timestamps": rbln_token_timestamps,
            }
        )

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls, compiled_models: List[rebel.RBLNCompiledModel], rbln_device_map: Dict[str, int]
    ) -> List[rebel.Runtime]:
        device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
        return [
            compiled_models[0].create_runtime("encoder", tensor_type="pt", device=device_val),
            compiled_models[0].create_runtime("decoder", tensor_type="pt", device=device_val),
        ]

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        whisper don't use attention_mask,
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
        """
        return {
            "input_ids": input_ids,
            "cache_position": cache_position,
        }

    # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L512
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, *args, **kwargs
    ) -> Dict[str, Any]:
        if not self.is_language_detected:
            model_kwargs["encoder_outputs"] = self.encoder(input_features=inputs_tensor)
            self.decoder_attention_mask = torch.zeros(self.batch_size, self.dec_max_seq_len, dtype=torch.int64)
        else:
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=torch.tensor([[-1.0]]))

        return model_kwargs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Seq2SeqLMOutput] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # default decoder pass
        if input_features is None and encoder_outputs is None:
            cross_attentions = []
            for step in cache_position:
                # skip step 0 if language_detection has been processed
                if step == 0 and self.is_language_detected:
                    cross_attentions.append(self.language_cross)
                    self.is_language_detected = False
                else:
                    self.decoder_attention_mask[:, step] = 1
                    decoder_output = self.decoder(
                        decoder_input_ids=input_ids[:, step : step + 1].contiguous(),
                        decoder_attention_mask=self.decoder_attention_mask,
                        cache_position=step.to(torch.int32),
                    )
                    cross_attentions.append(decoder_output.cross_attentions)
                    lm_logits = decoder_output.logits

            if self.rbln_token_timestamps:
                cross_attentions = torch.cat(cross_attentions, dim=-2)
            else:
                cross_attentions = None

            return Seq2SeqLMOutput(logits=lm_logits, cross_attentions=cross_attentions)

        # detect language pass
        # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/models/whisper/generation_whisper.py#L1442
        else:
            if encoder_outputs is None:
                self.encoder(input_features=input_features.contiguous())
            self.decoder_attention_mask = torch.zeros(self.batch_size, self.dec_max_seq_len, dtype=torch.int64)
            self.is_language_detected = True
            self.decoder_attention_mask[:, 0] = 1
            decoder_output = self.decoder(
                decoder_input_ids=decoder_input_ids.contiguous(),
                decoder_attention_mask=self.decoder_attention_mask,
                cache_position=torch.zeros([], dtype=torch.int32),
            )
            lm_logits = decoder_output.logits
            self.language_cross = decoder_output.cross_attentions
            return Seq2SeqLMOutput(logits=lm_logits)
