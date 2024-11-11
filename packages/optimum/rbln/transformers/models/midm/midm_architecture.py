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
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from ....transformers.models.decoderonly.decoderonly_architecture import (
    RotaryEmbedding,
    rotate_half,
    slice_and_unsqueeze_cos_sin,
)
from ...cache_utils import RebelDynamicCache_4D


def apply_rotary_to_tensor(tensor, cos, sin, rot_dim):
    """Applies rotary position embedding to the specified dimension of the tensor."""
    tensor_, tensor_pass = tensor[..., :rot_dim], tensor[..., rot_dim:]
    tensor_embed = (tensor_ * cos) + (rotate_half(tensor_) * sin)
    return torch.cat((tensor_embed, tensor_pass), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    rot_dim = cos.shape[-1]
    q_embed = apply_rotary_to_tensor(q, cos, sin, rot_dim)
    k_embed = apply_rotary_to_tensor(k, cos, sin, rot_dim)
    return q_embed, k_embed


class MidmLMHeadModelWrapper(torch.nn.Module):
    """A wrapper class for the Midm model with a language modeling head."""

    def __init__(self, model, max_seq_len):
        super().__init__()
        self.model = model.transformer
        self.lm_head = model.lm_head
        self.config = model.config
        self.head_dim = self.config.n_embd // self.config.n_head
        self.max_position_embeddings = (
            self.config.max_position_embeddings if max_seq_len > self.config.max_position_embeddings else max_seq_len
        )
        self.max_seq_len = max_seq_len
        self.rotary_dim = int(
            model.config.hidden_size // model.config.num_attention_heads * model.config.rotary_percentage
        )
        self.rotary_emb = self._init_rope()

    def _init_rope(self):
        """Initializes the Rotary Position Embeddings."""
        rotary_emb = RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
        )
        return rotary_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.LongTensor,
        batch_position: int,
        query_idx: int,
        *past_key_values,
    ):
        """Defines the forward pass for the wrapper model."""
        if input_ids.shape[1] == 1:
            rbln_batch_position = None
        else:
            rbln_batch_position = batch_position

        past_key_values = RebelDynamicCache_4D.from_input_format(
            cache_position,
            self.config.num_hidden_layers,
            *past_key_values,
        )

        outputs = _MidmModel.forward(
            self.model,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=cache_position,
            rotary_pos_emb=self.rotary_emb,
            batch_ids=rbln_batch_position,
        )

        hidden_states = outputs[0]
        if batch_position >= 0:
            hidden_states = hidden_states[:, query_idx].unsqueeze(1)

        logits = self.lm_head(hidden_states)
        output = (logits,) + outputs[1:]

        return output, batch_position + query_idx


def layernorm1p(module, input):
    """Applies Layer Normalization with a slight modification on the weights."""
    return torch.nn.functional.layer_norm(input, module.normalized_shape, module.weight + 1, module.bias, module.eps)


class _MidmAttention:
    """Custom implementation of the MidmAttention class with specific modifications."""

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Computes the attention weights and output."""
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if self.scale_attn_by_inverse_layer_idx or self.scale_qk_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if self.scale_qk_by_inverse_layer_idx:
            attn_weights = attn_weights * float(self.layer_idx + 1)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[RebelDynamicCache_4D] = None,
        batch_index: Optional[int] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """Defines the forward pass for the attention mechanism."""
        bsz, q_len, _ = hidden_states.size()

        querys, keys, values = self.c_attn(hidden_states).split(self.split_size, dim=2)

        querys = self._split_heads(querys, self.num_heads, self.head_dim).contiguous()
        keys = self._split_heads(keys, self.num_heads, self.head_dim).contiguous()
        values = self._split_heads(values, self.num_heads, self.head_dim).contiguous()

        querys, keys = apply_rotary_pos_emb(querys, keys, cos, sin)

        # Decoder
        if (batch_index is None or batch_index == -1) and bsz > 1:
            all_key_states = []
            all_value_states = []
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

                attn_output, _ = _MidmAttention._attn(self, query, key, value, attn_mask)
                attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

                all_key_states.append(key)
                all_value_states.append(value)
                all_attn_output.append(attn_output)

            keys = torch.cat(all_key_states, dim=0)
            values = torch.cat(all_value_states, dim=0)
            attn_output = torch.cat(all_attn_output, dim=0)

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

            attn_output, _ = _MidmAttention._attn(self, querys, keys, values, attention_mask)
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)
        return attn_output, keys, values


class _MidmBlock:
    """Custom implementation of the MidmBlock class with specific modifications."""

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_idx: int,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[RebelDynamicCache_4D] = None,
        batch_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        """Defines the forward pass for the block."""
        residual = hidden_states
        if self.use_layernorm1p:
            hidden_states = layernorm1p(self.ln_1, hidden_states)
        else:
            hidden_states = self.ln_1(hidden_states)

        hidden_states, k, v = _MidmAttention.forward(
            self.attn,
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cos=cos,
            sin=sin,
            batch_index=batch_ids,
        )
        past_key_value.assign(k, v, layer_idx)

        hidden_states = hidden_states + residual

        residual = hidden_states
        if self.use_layernorm1p:
            hidden_states = layernorm1p(self.ln_2, hidden_states)
        else:
            hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, past_key_value


class _MidmModel:
    """Custom implementation of the MidmModel class with specific modifications."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RebelDynamicCache_4D] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        rotary_pos_emb=None,
        batch_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Defines the forward pass for the model."""
        input_shape = input_ids.size()

        attention_mask = (1.0 - attention_mask) * -10000.0

        inputs_embeds = self.wte(input_ids)

        cos, sin = rotary_pos_emb(inputs_embeds, attention_mask.shape[-1])
        cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)
        hidden_states = inputs_embeds

        for layer_idx, (block, _) in enumerate(zip(self.h, past_key_values)):
            hidden_states, updated_cache = _MidmBlock.forward(
                block,
                hidden_states,
                layer_idx,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                batch_ids=batch_ids,
                cos=cos,
                sin=sin,
            )

        hidden_states = layernorm1p(self.ln_f, hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(output_shape)

        next_cache = updated_cache.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )
