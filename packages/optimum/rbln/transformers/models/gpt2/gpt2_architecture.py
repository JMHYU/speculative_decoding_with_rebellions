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

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...cache_utils import RebelDynamicCache_4D


class GPT2LMHeadModelWrapper(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.model = model.transformer
        self.lm_head = model.lm_head
        self.config = model.config
        self.max_seq_len = max_seq_len
        self.forward_dict = self.get_forward_dict()

    def get_forward_dict(self):
        forward_dict = {
            "wrapper": _GPT2Model.forward,
            "model": _GPT2Block.forward,
            "decoder_layer": _GPT2Attention.forward,
        }
        return forward_dict

    def forward(
        self,
        input_ids,
        attention_mask,
        cache_position,
        batch_position,
        query_idx,
        *past_key_values,
    ):
        if input_ids.shape[1] == 1:
            rbln_batch_position = None
        else:
            rbln_batch_position = batch_position

        # Formatting list of past_kv to DynamicCache class.
        past_key_value = RebelDynamicCache_4D.from_input_format(
            cache_position,
            self.config.n_layer,
            *past_key_values,
        )

        outputs = self.forward_dict["wrapper"](
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=cache_position,
            past_key_value=past_key_value,
            batch_ids=rbln_batch_position,
            forward_dict=self.forward_dict,
            # rotary_emb  differenct from_llama
        )

        hidden_states = outputs[0]
        if batch_position >= 0:
            hidden_states = hidden_states[:, query_idx].unsqueeze(1)
        logits = self.lm_head(hidden_states)

        output = (logits,) + outputs[1:]

        return output, batch_position + query_idx


class _GPT2Model:
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[RebelDynamicCache_4D] = None,
        batch_ids: Optional[torch.LongTensor] = None,
        forward_dict: Optional[Dict[str, classmethod]] = None,
    ) -> BaseModelOutputWithPast:
        b_size, q_len = input_ids.shape
        inputs_embeds = self.wte(input_ids)

        if position_ids.shape[0] > 1:
            position_embeds = []
            for b_idx in range(b_size):
                position_embed = self.wpe(position_ids[b_idx])
                # position_embed = position_embed.dtype(inputs_embeds.dtype)
                position_embeds.append(position_embed)

            position_embeds = torch.cat(position_embeds, dim=0).unsqueeze(1)
        else:
            position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds

        # GPT2Attention mask.
        # Here we assume mask is causal mask, (batch, 1, query_length, key_length + query_length)
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        for layer_idx, block in enumerate(self.h):
            hidden_states, updated_cache = forward_dict["model"](
                block,
                hidden_states,
                layer_idx,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                position_ids=position_ids,
                batch_ids=batch_ids,
                forward_dict=forward_dict,
            )

        hidden_states = self.ln_f(hidden_states)
        output_shape = (-1,) + (q_len,) + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(output_shape)

        # convert RebelDynamicCache to legacy Tuple[Tuple[torch.Tensor]]
        next_cache = updated_cache.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class _GPT2Block:
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_idx: int,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[RebelDynamicCache_4D] = None,
        batch_ids: Optional[torch.LongTensor] = None,
        forward_dict: Optional[Dict[str, classmethod]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, RebelDynamicCache_4D]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states, k, v = forward_dict["decoder_layer"](
            self.attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_ids,
        )
        past_key_value.assign(k, v, layer_idx)

        # residual connection
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class _GPT2Attention:
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # -------------------
        # Below are deleted since "where" op does not supported on RBLN graph.
        # -------------------
        # if not self.is_cross_attention:
        #     # if only "normal" attention layer implements causal mask
        #     query_length, key_length = query.size(-2), key.size(-2)
        #     causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        #     mask_value = torch.finfo(attn_weights.dtype).min
        #     # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        #     # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        #     mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        #     attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        # Apply the attention mask
        attn_weights.view(
            -1,
        )
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[RebelDynamicCache_4D] = None,
        batch_index: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        bsz, q_len, _ = hidden_states.size()
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        querys = self._split_heads(query, self.num_heads, self.head_dim)  # (batch, head, seq_length, head_features)
        keys = self._split_heads(key, self.num_heads, self.head_dim)
        values = self._split_heads(value, self.num_heads, self.head_dim)

        # Decoder
        if (batch_index is None or batch_index == -1) and bsz > 1:
            all_keys = []
            all_values = []
            all_attn_output = []

            for b in range(bsz):
                query = querys[b].unsqueeze(0)
                attn_mask = attention_mask[b].unsqueeze(0)
                key = keys[b].unsqueeze(0)
                value = values[b].unsqueeze(0)

                key, value = past_key_value.update(
                    key,
                    value,
                    self.layer_idx,
                    b,
                )

                attn_output, _ = _GPT2Attention._attn(self, query, key, value, attn_mask)
                attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

                all_keys.append(key)
                all_values.append(value)
                all_attn_output.append(attn_output)

            keys = torch.cat(all_keys, dim=0)
            values = torch.cat(all_values, dim=0)
            attn_output = torch.cat(all_attn_output, dim=0)

        # Prefill
        else:
            if batch_index is None or batch_index == -1:
                batch_index = 0

            keys, values = past_key_value.update(
                keys,
                values,
                self.layer_idx,
                batch_index,
                read_first_step=True,
            )

            attn_output, _ = _GPT2Attention._attn(self, querys, keys, values, attention_mask)
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)

        return attn_output, keys, values
