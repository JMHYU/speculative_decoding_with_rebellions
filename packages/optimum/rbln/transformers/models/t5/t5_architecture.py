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

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5LayerCrossAttention,
    T5LayerSelfAttention,
    T5Stack,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import T5ForConditionalGeneration


class T5Encoder(T5Stack):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
        batch_ids: torch.Tensor = None,
    ) -> BaseModelOutput:
        hidden_states = self.embed_tokens(input_ids)
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        position_bias = position_bias + extended_attention_mask
        for i, layer_module in enumerate(self.block):
            layer_outputs = _T5Block.forward(
                layer_module,
                hidden_states,
                position_bias=position_bias,
                batch_ids=batch_ids,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class T5Decoder(T5Stack):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        past_key_values: torch.Tensor,
        position_bias: torch.Tensor,
        encoder_decoder_position_bias: torch.Tensor,
        cache_position: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        hidden_states = self.embed_tokens(input_ids)
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = encoder_decoder_position_bias + encoder_extended_attention_mask

        present_key_value_states = ()

        for layer_module, past_key_value in zip(self.block, past_key_values):
            layer_outputs = _T5Block.forward(
                layer_module,
                hidden_states,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                past_key_value=past_key_value,
                cache_position=cache_position,
                batch_ids=batch_ids,
            )
            hidden_states, present_key_value_state = layer_outputs[:2]
            present_key_value_states = present_key_value_states + (present_key_value_state,)

        hidden_states = self.final_layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
        )


class T5EncoderWrapper(torch.nn.Module):
    def __init__(self, model: "T5ForConditionalGeneration"):
        super().__init__()
        self.config = model.config
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.default_max_length = getattr(self.config, "n_positions", None) or getattr(
            self.config, "max_position_embeddings", None
        )
        self.encoder_max_length = None
        self.decoder_max_length = None
        self.decoder_batch_size = 1

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                cross_key_value: torch.Tensor = None, batch_idx: torch.Tensor = None,
                ) -> torch.Tensor:
        encoder_batch_size = input_ids.shape[0]
        decoder_batch_size = self.decoder_batch_size
        decoder_max_length = self.decoder_max_length or self.default_max_length
        encoder_max_length = self.encoder_max_length or self.default_max_length

        attn_layer = self.encoder.block[0].layer[0].SelfAttention
        encoder_position_bias = T5Attention.compute_bias(attn_layer, encoder_max_length, encoder_max_length)
        encoder_outputs = T5Encoder.forward(self.encoder, input_ids, attention_mask, encoder_position_bias, batch_ids=torch.tensor(0, dtype=torch.int32))

        attn_layer = self.decoder.block[0].layer[0].SelfAttention
        decoder_position_bias = T5Attention.compute_bias(attn_layer, decoder_max_length, decoder_max_length)
        decoder_position_bias = decoder_position_bias[:, :, :1]

        attn_layer = self.decoder.block[0].layer[1].EncDecAttention
        encoder_decoder_position_bias = torch.zeros(1, attn_layer.n_heads, 1, encoder_max_length)

        dummy_past_key_value = []
        for i in range(self.config.num_layers):
            pkv_self_attn_key = torch.zeros(
                decoder_batch_size, self.config.num_heads, decoder_max_length, self.config.d_kv
            )
            pkv_self_attn_value = torch.zeros(
                decoder_batch_size, self.config.num_heads, decoder_max_length, self.config.d_kv
            )
            pkv_cross_attn_key = torch.zeros(
                encoder_batch_size, self.config.num_heads, encoder_max_length, self.config.d_kv
            )
            pkv_cross_attn_value = torch.zeros(
                encoder_batch_size, self.config.num_heads, encoder_max_length, self.config.d_kv
            )
            layer_pkv = (pkv_self_attn_key, pkv_self_attn_value, pkv_cross_attn_key, pkv_cross_attn_value)
            dummy_past_key_value.append(layer_pkv)

        decoder_attention_mask = torch.zeros(decoder_batch_size, decoder_max_length, dtype=torch.float32)
        decoder_attention_mask[:, :1] = 1

        # Since first step of decoder has different graph to further step of it,
        # here we merges decoder into its corresponding encoder.
        # TODO(jongho): Separate first-step-decoder.
        decoder_outputs = T5Decoder.forward(
            self.decoder,
            input_ids=torch.zeros(decoder_batch_size, 1, dtype=torch.int64),
            attention_mask=decoder_attention_mask,
            position_bias=decoder_position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=dummy_past_key_value,
            cache_position=torch.tensor(0, dtype=torch.int32),
            batch_ids=torch.tensor(0, dtype=torch.int32),
        )

        past_key_values = decoder_outputs.past_key_values

        cross_kv_cache = []
        for i in range(self.model.config.num_layers):
            cross_kv_cache.append(past_key_values[i][2])
            cross_kv_cache.append(past_key_values[i][3])
        cross_kv_cache = torch.stack(cross_kv_cache, dim=0)

        cross_key_value = cross_key_value.slice_scatter(cross_kv_cache, dim=1, start=batch_idx, end=batch_idx+1)

        return cross_key_value


class T5DecoderWrapper(torch.nn.Module):
    def __init__(self, model: "T5ForConditionalGeneration"):
        super().__init__()
        self.config = model.config
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.default_max_length = getattr(self.config, "n_positions", None) or getattr(
            self.config, "max_position_embeddings", None
        )
        self.encoder_max_length = None
        self.decoder_max_length = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        batch_position: torch.Tensor,
        self_kv_cache: torch.Tensor,
        cross_kv_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # cache_position : step 0부터
        # attention_mask : 1개가 색칠된것부터 ([0:cache_position+1])
        num_layers = self.model.config.num_layers
        encoder_max_length = self.encoder_max_length or self.default_max_length
        decoder_max_length = self.decoder_max_length or self.default_max_length

        if input_ids.shape[1] == 1:
            rbln_batch_position = None
        else:
            rbln_batch_position = batch_position

        kv_cache = ()
        for i in range(0, num_layers * 2, 2):
            kv_cache = kv_cache + (
                (
                    self_kv_cache[i],
                    self_kv_cache[i + 1],
                    cross_kv_cache[i],
                    cross_kv_cache[i + 1],
                ),
            )

        attn_layer = self.model.decoder.block[0].layer[0].SelfAttention
        _decoder_position_bias = T5Attention.compute_bias(attn_layer, decoder_max_length, decoder_max_length)

        batch_decoder_position_bias = []
        for i in range(input_ids.shape[0]):
            batch_position_bias = _decoder_position_bias[:, :, cache_position[i][0]].unsqueeze(2)
            batch_decoder_position_bias.append(batch_position_bias)
        decoder_position_bias = torch.cat(batch_decoder_position_bias, dim=0)

        attn_layer = self.model.decoder.block[0].layer[1].EncDecAttention
        encoder_decoder_position_bias = torch.zeros(1, attn_layer.n_heads, 1, encoder_max_length)

        decoder_outputs = T5Decoder.forward(
            self.model.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=1,
            encoder_attention_mask=encoder_attention_mask,
            position_bias=decoder_position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            past_key_values=kv_cache,
            cache_position=cache_position,
            batch_ids=rbln_batch_position
        )

        past_key_values = decoder_outputs.past_key_values
        sequence_output = decoder_outputs[0]
        if self.model.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model.model_dim**-0.5)
        lm_logits = self.model.lm_head(sequence_output)

        self_kv_cache = []
        for i in range(self.model.config.num_layers):
            self_kv_cache.append(past_key_values[i][0])
            self_kv_cache.append(past_key_values[i][1])

        self_kv_cache = torch.stack(self_kv_cache, dim=0)

        return lm_logits, self_kv_cache, batch_position


class _T5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Tuple[torch.Tensor] = None,
        position_bias: torch.Tensor = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,  # 현재 cache sequence 길이
        batch_index: torch.Tensor = None,
        is_self_attn: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        batch_size = hidden_states.shape[0]

        def shape(states, batch_size):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states, batch_size):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        query_states = shape(self.q(hidden_states), batch_size)  # (batch_size, n_heads, seq_length, dim_per_head)

        # projection
        if is_self_attn:
            key_states = shape(self.k(hidden_states), batch_size)
            value_states = shape(self.v(hidden_states), batch_size)
        else:
            # cross-attn
            if cache_position.dim() == 0 :
                key_states = shape(self.k(key_value_states), key_value_states.shape[0])
                value_states = shape(self.v(key_value_states), key_value_states.shape[0])
                past_key_value = key_states, value_states
            else:
                key_states = past_key_value[0]
                value_states = past_key_value[1]

        if (batch_index is None or batch_index == -1) and batch_size > 1:
            all_key_states = []
            all_value_states = []
            all_attn_output = []

            for b in range(batch_size):
                batch_query_states = query_states[b].unsqueeze(0)
                batch_key_states = key_states[b].unsqueeze(0)
                batch_value_states = value_states[b].unsqueeze(0)

                if is_self_attn and past_key_value is not None:
                    batch_key_states = past_key_value[0][b].unsqueeze(0).slice_scatter(
                        batch_key_states, dim=-2, start=cache_position[b][0], end=cache_position[b][0] + 1
                    )
                    batch_value_states = past_key_value[1][b].unsqueeze(0).slice_scatter(
                        batch_value_states, dim=-2, start=cache_position[b][0], end=cache_position[b][0] + 1
                    )

                scores = torch.matmul(batch_query_states, batch_key_states.transpose(3, 2))
                scores += position_bias[b]
                attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                    scores
                )
                attn_output = unshape(torch.matmul(attn_weights, batch_value_states), 1)
                all_key_states.append(batch_key_states)
                all_value_states.append(batch_value_states)
                all_attn_output.append(attn_output)

            key_states = torch.cat(all_key_states, dim=0)
            value_states = torch.cat(all_value_states, dim=0)
            attn_output = torch.cat(all_attn_output, dim=0)

        else:
            if batch_index is None or batch_index == -1:
                batch_index = 0

            if is_self_attn and past_key_value is not None:
                key_states = past_key_value[0].slice_scatter(
                    key_states, dim=2, start=cache_position, end=cache_position + 1
                )
                value_states = past_key_value[1].slice_scatter(
                    value_states, dim=2, start=cache_position, end=cache_position + 1
                )
            # compute scores
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            scores += position_bias

            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)

            attn_output = unshape(torch.matmul(attn_weights, value_states), batch_size)  # (batch_size, seq_length, dim)

        attn_output = self.o(attn_output)
        present_key_value = (key_states, value_states)
        outputs = (attn_output,) + (present_key_value,)
        return outputs


class _T5LayerSelfAttention(T5LayerSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor = None,
        past_key_value: Tuple[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        batch_index: torch.Tensor = None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = _T5Attention.forward(
            self.SelfAttention,
            hidden_states=normed_hidden_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            cache_position=cache_position,
            batch_index=batch_index,
            is_self_attn=True,
        )

        # Residual Connection
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class _T5LayerCrossAttention(T5LayerCrossAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        position_bias: torch.Tensor = None,
        past_key_value: Tuple[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        batch_index: torch.Tensor = None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = _T5Attention.forward(
            self.EncDecAttention,
            hidden_states=normed_hidden_states,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            cache_position=cache_position,
            batch_index=batch_index,
            is_self_attn=False,
        )

        # Residual connection
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class _T5Block(T5Block):
    def forward(
        self,
        hidden_states,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_decoder_position_bias=None,
        past_key_value=None,
        cache_position=None,
        batch_ids=None,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            if self_attn_past_key_value == (None, None):
                self_attn_past_key_value = None

            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        self_attention_outputs = _T5LayerSelfAttention.forward(
            self.layer[0],
            hidden_states=hidden_states,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
            cache_position=cache_position,
            batch_index=batch_ids,
        )

        hidden_states, present_key_value_state = self_attention_outputs[:2]

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = _T5LayerCrossAttention.forward(
                self.layer[1],
                hidden_states,
                key_value_states=encoder_hidden_states,
                position_bias=encoder_decoder_position_bias,
                past_key_value=cross_attn_past_key_value,
                cache_position=cache_position,
                batch_index=batch_ids,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                # print(present_key_value_state.shape)
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        outputs = (hidden_states,)
        outputs = outputs + (present_key_value_state,)

        return outputs
