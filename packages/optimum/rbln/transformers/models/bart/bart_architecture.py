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

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartDecoder,
    BartDecoderLayer,
    BartForConditionalGeneration,
    BartSdpaAttention,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class _BartAttention(BartAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        batch_index: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
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
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if cache_position.dim() > 0:
            proj_shape = (bsz, self.num_heads, -1, self.head_dim)
            query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
            key_states = key_states.reshape(*proj_shape)
            value_states = value_states.reshape(*proj_shape)

            all_key_states = []
            all_value_states = []
            all_attn_output = []
            for b in range(bsz):
                batch_query_states = query_states[b].unsqueeze(0).unsqueeze(2)
                batch_attention_mask = attention_mask[b].unsqueeze(0).unsqueeze(2)
                batch_key_states = key_states[b].unsqueeze(0).unsqueeze(2)
                batch_value_states = value_states[b].unsqueeze(0).unsqueeze(2)
                if not is_cross_attention:
                    batch_key_states = (
                        past_key_value[0][b]
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .slice_scatter(
                            batch_key_states, dim=-2, start=cache_position[b][0], end=cache_position[b][0] + 1
                        )
                    )
                    batch_value_states = (
                        past_key_value[1][b]
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .slice_scatter(
                            batch_value_states, dim=-2, start=cache_position[b][0], end=cache_position[b][0] + 1
                        )
                    )
                attn_weights = torch.matmul(batch_query_states, batch_key_states.transpose(3, 4))
                attn_weights = attn_weights + batch_attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                attn_output = torch.matmul(attn_weights, batch_value_states)
                attn_output = attn_output.view(1, self.num_heads, tgt_len, self.head_dim)
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(1, tgt_len, self.embed_dim)
                all_key_states.append(batch_key_states)
                all_value_states.append(batch_value_states)
                all_attn_output.append(attn_output)
            key_states = torch.cat(all_key_states, dim=0).squeeze(2)
            value_states = torch.cat(all_value_states, dim=0).squeeze(2)
            attn_output = torch.cat(all_attn_output, dim=0)

        else:
            if batch_index is None or batch_index == -1:
                batch_index = 0

            if not is_cross_attention:
                key_states = past_key_value[0].slice_scatter(
                    key_states, dim=2, start=cache_position, end=cache_position + 1
                )
                value_states = past_key_value[1].slice_scatter(
                    value_states, dim=2, start=cache_position, end=cache_position + 1
                )

            proj_shape = (bsz * self.num_heads, -1, self.head_dim)
            query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
            key_states = key_states.reshape(*proj_shape)
            value_states = value_states.reshape(*proj_shape)

            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            attn_output = torch.bmm(attn_weights, value_states)
            attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2)
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        present_key_value = (key_states, value_states)

        return attn_output, present_key_value


class _BartSdpaAttention(BartSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        batch_index: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
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
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        query_states = self._shape(query_states, tgt_len, bsz)

        if (batch_index is None or batch_index == -1) and bsz > 1:
            all_key_states = []
            all_value_states = []
            all_attn_output = []

            for b in range(bsz):
                batch_query_states = query_states[b].unsqueeze(0)
                batch_attention_mask = attention_mask[b].unsqueeze(0)
                batch_key_states = key_states[b].unsqueeze(0)
                batch_value_states = value_states[b].unsqueeze(0)

                if not is_cross_attention:
                    batch_key_states = (
                        past_key_value[0][b]
                        .unsqueeze(0)
                        .slice_scatter(
                            batch_key_states, dim=-2, start=cache_position[b][0], end=cache_position[b][0] + 1
                        )
                    )
                    batch_value_states = (
                        past_key_value[1][b]
                        .unsqueeze(0)
                        .slice_scatter(
                            batch_value_states, dim=-2, start=cache_position[b][0], end=cache_position[b][0] + 1
                        )
                    )

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    batch_query_states, batch_key_states, batch_value_states, attn_mask=batch_attention_mask
                )
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(1, tgt_len, self.embed_dim)
                all_key_states.append(batch_key_states)
                all_value_states.append(batch_value_states)
                all_attn_output.append(attn_output)

            key_states = torch.cat(all_key_states, dim=0)
            value_states = torch.cat(all_value_states, dim=0)
            attn_output = torch.cat(all_attn_output, dim=0)

        else:
            if batch_index is None or batch_index == -1:
                batch_index = 0

            if not is_cross_attention:
                key_states = past_key_value[0].slice_scatter(
                    key_states, dim=2, start=cache_position, end=cache_position + 1
                )
                value_states = past_key_value[1].slice_scatter(
                    value_states, dim=2, start=cache_position, end=cache_position + 1
                )

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
            )
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        present_key_value = (key_states, value_states)

        return attn_output, present_key_value


ATTN_FORWARD_MAP = {"eager": _BartAttention.forward, "sdpa": _BartSdpaAttention.forward}


class _BartDecoderLayer(BartDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        cache_position: torch.Tensor,
        batch_ids: torch.Tensor,
        attn_impl: str = "eager",
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # Self Attention Block
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2]

        hidden_states, present_key_value = ATTN_FORWARD_MAP[attn_impl](
            self.self_attn,
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            cache_position=cache_position,
            batch_index=batch_ids,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        residual = hidden_states
        cross_attn_past_key_value = past_key_value[-2:]

        hidden_states, cross_attn_present_key_value = ATTN_FORWARD_MAP[attn_impl](
            self.encoder_attn,
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_attn_past_key_value,
            attention_mask=encoder_attention_mask,
            cache_position=cache_position,
            batch_index=batch_ids,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected Block
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, present_key_value


class _BartDecoder(BartDecoder):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: torch.Tensor,
        cache_position: torch.Tensor,
        batch_ids: torch.Tensor,
        attn_impl: str = "eager",
    ):
        # embedding
        # thkim fix : transformers == 4.44.2 compile
        if hasattr(self, "embed_scale"):
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position.dim() == 0:
            positions_idx = cache_position + self.embed_positions.offset
            positions = self.embed_positions.weight[positions_idx]
            hidden_states = inputs_embeds + positions
        else:
            hidden_all = []
            for i in range(input_ids.shape[0]):
                # cache position [N,1]
                positions_idx = cache_position[i]
                position_weight = self.embed_positions.weight[2:]
                position = position_weight[positions_idx]
                tmp_hidden = position + inputs_embeds[i]
                hidden_all.append(tmp_hidden)
            hidden_states = torch.stack(hidden_all, dim=0)

        hidden_states = self.layernorm_embedding(hidden_states)

        # prepare attn_mask
        input_shape = input_ids.size()
        if self._use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, cache_position
            )
            encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                encoder_attention_mask, torch.float32, tgt_len=input_shape[-1]
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, cache_position
            )
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, torch.float32, tgt_len=input_shape[-1]
            )

        # iterate decoder_layer
        next_decoder_cache = ()
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            layer_outputs = _BartDecoderLayer.forward(
                decoder_layer,
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                batch_ids=batch_ids,
                attn_impl=attn_impl,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
        )


class BartDecoderWrapper(torch.nn.Module):
    def __init__(self, model: "BartForConditionalGeneration"):
        super().__init__()
        self.config = model.config
        self.decoder = model.get_decoder()
        self.num_layers = self.config.decoder_layers
        self.lm_head = model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        batch_position: torch.Tensor,
        self_kv_cache: torch.Tensor,
        cross_kv_cache: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor]]:
        if input_ids.shape[1] == 1:
            rbln_batch_position = None
        else:
            rbln_batch_position = batch_position
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
        # decode
        decoder_outputs = _BartDecoder.forward(
            self.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            cache_position=cache_position,
            past_key_values=kv_cache,
            encoder_hidden_states=torch.tensor([1]),
            attn_impl=self.config._attn_implementation,
            batch_ids=rbln_batch_position,
        )
        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        # get self_kv_cache from ouputs
        past_key_values = decoder_outputs[1]
        self_kv_cache = []
        for i in range(self.num_layers):
            self_kv_cache.append(past_key_values[i][0])
            self_kv_cache.append(past_key_values[i][1])
        self_kv_cache = torch.stack(self_kv_cache, dim=0)

        return lm_logits, self_kv_cache, batch_position


class BartEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.decoder = model.get_decoder()
        self.encoder = model.get_encoder()
        self.num_layers = self.config.encoder_layers
        self.decoder_max_length = self.config.max_position_embeddings
        self.encoder_max_length = self.config.max_position_embeddings
        self.num_heads = self.config.decoder_attention_heads
        self.d_kv = self.config.d_model // self.num_heads

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        cross_key_value: torch.Tensor = None,
        batch_idx: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        encoder_batch_size = input_ids.shape[0]
        decoder_batch_size = encoder_batch_size  # TODO(taehoon) fix to enable beam-search

        # 1. run encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = encoder_outputs[0]

        # 2. run dummy decoder to get pre-calculated cross-key_values for generation
        dummy_past_key_value = []
        for _ in range(self.num_layers):
            pkv_self_attn_key = torch.zeros(decoder_batch_size, self.num_heads, self.decoder_max_length, self.d_kv)
            pkv_self_attn_value = torch.zeros(decoder_batch_size, self.num_heads, self.decoder_max_length, self.d_kv)
            pkv_cross_attn_key = torch.zeros(encoder_batch_size, self.num_heads, self.encoder_max_length, self.d_kv)
            pkv_cross_attn_value = torch.zeros(encoder_batch_size, self.num_heads, self.encoder_max_length, self.d_kv)
            layer_pkv = (pkv_self_attn_key, pkv_self_attn_value, pkv_cross_attn_key, pkv_cross_attn_value)
            dummy_past_key_value.append(layer_pkv)

        decoder_attention_mask = torch.zeros(decoder_batch_size, self.decoder_max_length, dtype=torch.float32)
        decoder_attention_mask[:, :1] = 1

        decoder_outputs = _BartDecoder.forward(
            self.decoder,
            input_ids=torch.zeros((decoder_batch_size, 1), dtype=torch.int64),
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            cache_position=torch.tensor(0, dtype=torch.int32),
            encoder_hidden_states=last_hidden_states,
            past_key_values=dummy_past_key_value,
            batch_ids=torch.tensor(0, dtype=torch.int32),
            attn_impl=self.config._attn_implementation,
        )
        first_past_kv = decoder_outputs[1]

        encoder_kv = []
        for i in range(self.model.config.decoder_layers):
            encoder_kv.append(first_past_kv[i][2].unsqueeze(0))
            encoder_kv.append(first_past_kv[i][3].unsqueeze(0))
        encoder_kv = torch.cat(encoder_kv, dim=0)

        cross_key_value = cross_key_value.slice_scatter(encoder_kv, dim=1, start=batch_idx, end=batch_idx + 1)

        return cross_key_value
