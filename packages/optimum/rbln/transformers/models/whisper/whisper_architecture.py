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

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperPositionalEmbedding,
    WhisperSdpaAttention,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class _WhisperAttention(WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, _ = hidden_states.size()
        is_cross_attention = key_value_states is not None

        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention:
            is_dummy_decoder = len(key_value_states.shape) > 1
            if is_dummy_decoder:
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            else:
                key_states = past_key_value[0]
                value_states = past_key_value[1]
        else:
            if self.is_decoder:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
                key_states = past_key_value[0].slice_scatter(
                    key_states, dim=2, start=cache_position, end=cache_position + 1
                )
                value_states = past_key_value[1].slice_scatter(
                    value_states, dim=2, start=cache_position, end=cache_position + 1
                )
            else:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        src_len = key_states.size(1)
        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        return attn_output, attn_weights, present_key_value


class _WhisperSdpaAttention(WhisperSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, _ = hidden_states.size()

        is_cross_attention = key_value_states is not None

        query_states = self.q_proj(hidden_states)

        if is_cross_attention:
            is_dummy_decoder = len(key_value_states.shape) > 1
            if is_dummy_decoder:
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            else:
                key_states = past_key_value[0]
                value_states = past_key_value[1]
        else:
            if self.is_decoder:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
                key_states = past_key_value[0].slice_scatter(
                    key_states, dim=2, start=cache_position, end=cache_position + 1
                )
                value_states = past_key_value[1].slice_scatter(
                    value_states, dim=2, start=cache_position, end=cache_position + 1
                )
            else:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        query_states = self._shape(query_states, tgt_len, bsz)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.is_causal and attention_mask is None and tgt_len > 1,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, present_key_value


ATTN_FORWARD_MAP = {"eager": _WhisperAttention.forward, "sdpa": _WhisperSdpaAttention.forward}


class _WhisperDecoderLayer(WhisperDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        attn_impl: str = "eager",
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Self Attention Block
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        hidden_states, _, present_key_value = ATTN_FORWARD_MAP[attn_impl](
            self.self_attn,
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        cross_attn_past_key_value = past_key_value[2:] if past_key_value is not None else None
        if output_attentions:
            hidden_states, cross_attn_weights, cross_attn_present_key_value = _WhisperAttention.forward(
                self.encoder_attn,
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past_key_value,
                cache_position=cache_position,
            )
        else:
            hidden_states, cross_attn_weights, cross_attn_present_key_value = ATTN_FORWARD_MAP[attn_impl](
                self.encoder_attn,
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past_key_value,
                cache_position=cache_position,
            )
        hidden_states = residual + hidden_states
        present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected Block
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, cross_attn_weights


class _WhisperPositionalEmbedding(WhisperPositionalEmbedding):
    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]
        else:
            return self.weight[position_ids]


class _WhisperDecoder(WhisperDecoder):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attn_impl: str = "eager",
        output_attentions: bool = False,
        **kwargs,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # positional embeding
        inputs_embeds = self.embed_tokens(input_ids)
        positions = _WhisperPositionalEmbedding.forward(
            self.embed_positions, input_ids, cache_position, cache_position
        )
        hidden_states = inputs_embeds + positions

        # prepare casual_attn_mask
        if self._use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, cache_position
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, cache_position
            )

        next_decoder_cache = ()
        all_cross_attentions = () if output_attentions else None
        # iterate decoder_layer
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = _WhisperDecoderLayer.forward(
                decoder_layer,
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_key_value,
                cache_position=cache_position,
                attn_impl=attn_impl,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)
            if output_attentions:
                all_cross_attentions += (layer_outputs[2],)

        # layer_norm
        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            cross_attentions=all_cross_attentions,
        )


class _WhisperDecoderWrapper(torch.nn.Module):
    def __init__(self, model, output_attentions: bool = False):
        super().__init__()
        self.proj_out = model.proj_out
        self.config = model.config
        self.decoder = model.get_decoder()
        self.num_layers = self.config.decoder_layers
        self.attn_impl = self.config._attn_implementation
        self.output_attentions = output_attentions

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        self_kv_cache: torch.Tensor,
        cross_kv_cache: torch.Tensor,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        # prepare past_key_values
        kv_cache = ()
        for i in range(0, self.num_layers * 2, 2):
            kv_cache = kv_cache + (
                (
                    self_kv_cache[i],
                    self_kv_cache[i + 1],
                    cross_kv_cache[i],
                    cross_kv_cache[i + 1],
                ),
            )

        # Decode
        decoder_outputs = _WhisperDecoder.forward(
            self.decoder,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            cache_position=cache_position,
            past_key_values=kv_cache,
            encoder_hidden_states=torch.tensor([1]),
            attn_impl=self.attn_impl,
            output_attentions=self.output_attentions,
        )
        sequence_output = decoder_outputs[0]
        lm_logits = self.proj_out(sequence_output)

        # get self_kv_cache from ouputs
        past_key_values = decoder_outputs[1]
        self_kv_cache = []
        for i in range(self.config.decoder_layers):
            self_kv_cache.append(past_key_values[i][0])
            self_kv_cache.append(past_key_values[i][1])
        self_kv_cache = torch.stack(self_kv_cache, dim=0)

        if self.output_attentions:
            # deocder's cross attention is used for token_timestamps
            cross_attention = torch.stack(decoder_outputs[2], dim=0)
            return lm_logits, self_kv_cache, cross_attention
        else:
            return lm_logits, self_kv_cache


class _WhisperEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.decoder = model.get_decoder()
        self.encoder = model.get_encoder()
        self.num_layers = self.config.decoder_layers
        self.decoder_max_length = self.config.max_target_positions
        self.encoder_max_length = self.config.max_source_positions
        self.num_heads = self.config.decoder_attention_heads
        self.d_kv = self.config.d_model // self.num_heads
        self.attn_impl = self.config._attn_implementation

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        encoder_outputs = self.encoder(input_features=input_features)

        last_hidden_states = encoder_outputs[0]

        encoder_batch_size = input_features.shape[0]
        decoder_batch_size = encoder_batch_size  # TODO fix in future

        dummy_past_key_value = []
        for _ in range(self.num_layers):
            pkv_self_attn_key = torch.zeros(decoder_batch_size, self.num_heads, self.decoder_max_length, self.d_kv)
            pkv_self_attn_value = torch.zeros(decoder_batch_size, self.num_heads, self.decoder_max_length, self.d_kv)
            pkv_cross_attn_key = torch.zeros(encoder_batch_size, self.num_heads, self.encoder_max_length, self.d_kv)
            pkv_cross_attn_value = torch.zeros(encoder_batch_size, self.num_heads, self.encoder_max_length, self.d_kv)
            layer_pkv = (pkv_self_attn_key, pkv_self_attn_value, pkv_cross_attn_key, pkv_cross_attn_value)
            dummy_past_key_value.append(layer_pkv)

        decoder_attention_mask = torch.zeros(decoder_batch_size, self.decoder_max_length, dtype=torch.int64)
        decoder_attention_mask[:, :1] = 1

        decoder_outputs = _WhisperDecoder.forward(
            self.decoder,
            input_ids=torch.zeros((decoder_batch_size, 1), dtype=torch.int64),
            attention_mask=decoder_attention_mask,
            cache_position=torch.tensor(0, dtype=torch.int32),
            encoder_hidden_states=last_hidden_states,
            past_key_values=dummy_past_key_value,
            attn_impl=self.attn_impl,
            output_attentions=False,
        )

        first_past_kv = decoder_outputs[1]

        cross_kv = []
        for layer_out in first_past_kv:  # for layer
            cross_kv.append(layer_out[2])
            cross_kv.append(layer_out[3])
        cross_kv = torch.stack(cross_kv, dim=0)

        return cross_kv
