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

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)

from ...cache_utils import RebelDynamicCache
from ..decoderonly import (
    DecoderOnlyWrapper,
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    slice_and_unsqueeze_cos_sin,
)


class PhiWrapper(DecoderOnlyWrapper):
    def _init_rope(self):
        if self.rope_scaling is None:
            rotary_emb = RotaryEmbedding(
                int(self.config.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "linear":
                rotary_emb = LinearScalingRotaryEmbedding(
                    int(self.config.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.config.rope_theta,
                    max_seq_len=self.max_seq_len,
                )
            elif scaling_type == "dynamic":
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    int(self.config.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.config.rope_theta,
                    max_seq_len=self.max_seq_len,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

        return rotary_emb

    def get_forward_dict(self):
        forward_dict = {}
        forward_dict.update(
            {
                "wrapper": PhiModel.forward,
                "model": PhiDecoderLayer.forward,
                "decoder_layer": PhiAttention.forward,
            }
        )
        return forward_dict


class PhiAttention:
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[RebelDynamicCache] = None,
        batch_index: Optional[int] = None,
        output_attentions: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        rotary_pos_emb=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : rotary_pos_emb.dim],
            query_states[..., rotary_pos_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : rotary_pos_emb.dim],
            key_states[..., rotary_pos_emb.dim :],
        )

        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        # Decoder
        if (batch_index is None or batch_index == -1) and bsz > 1:
            all_key_states = []
            all_value_states = []
            all_attn_output = []

            for b in range(bsz):
                query_state = query_states[b].unsqueeze(0)
                attn_mask = attention_mask[b].unsqueeze(0)
                key_state = key_states[b].unsqueeze(0)
                value_state = value_states[b].unsqueeze(0)

                # reshape for removing repeat_kv (batch=1 , num_head, 1, q_len=1, head_dim)
                key_state = key_state.unsqueeze(2)
                value_state = value_state.unsqueeze(2)
                attn_mask = attn_mask.unsqueeze(2)

                query_state = query_state.view(
                    1,
                    self.num_key_value_heads,
                    self.num_heads // self.num_key_value_heads,
                    q_len,
                    self.head_dim,
                )

                key_state, value_state = past_key_value.update(
                    key_state,
                    value_state,
                    self.layer_idx,
                    b,
                )

                # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
                attn_weights = torch.matmul(
                    query_state.to(torch.float32),
                    key_state.to(torch.float32).transpose(3, 4),
                ) / math.sqrt(self.head_dim)
                attn_weights = attn_weights + attn_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_weights = nn.functional.dropout(
                    attn_weights, p=self.attention_dropout, training=self.training
                )
                attn_output = torch.matmul(attn_weights, value_state)

                # reshape for removing repeat_kv
                attn_output = attn_output.view(1, self.num_heads, q_len, self.head_dim)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(
                    1, q_len, self.num_heads * self.head_dim
                )

                all_key_states.append(key_state)
                all_value_states.append(value_state)
                all_attn_output.append(attn_output)

            key_states = torch.cat(all_key_states, dim=0)
            value_states = torch.cat(all_value_states, dim=0)
            attn_output = torch.cat(all_attn_output, dim=0)
        else:
            if batch_index is None or batch_index == -1:
                batch_index = 0

            # reshape for removing repeat_kv
            key_states = key_states.unsqueeze(2)
            value_states = value_states.unsqueeze(2)
            attention_mask = attention_mask.unsqueeze(2)
            query_states = query_states.view(
                1,
                self.num_key_value_heads,
                self.num_heads // self.num_key_value_heads,
                q_len,
                self.head_dim,
            )

            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                batch_index,
                read_first_step=True,
            )

            # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
            attn_weights = torch.matmul(
                query_states.to(torch.float32),
                key_states.to(torch.float32).transpose(3, 4),
            ) / math.sqrt(self.head_dim)
            attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(value_states.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, value_states)

            # reshape for removing repeat_kv
            attn_output = attn_output.view(1, self.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.head_dim
            )

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, key_states, value_states


class PhiDecoderLayer:
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[RebelDynamicCache] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        batch_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        rotary_pos_emb=None,
        forward_dict: Optional[Dict[str, classmethod]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, key_states, value_states = forward_dict[
            "decoder_layer"
        ](
            self.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            batch_index=batch_ids,
            use_cache=use_cache,
            cos=cos,
            sin=sin,
            rotary_pos_emb=rotary_pos_emb,
            **kwargs,
        )
        past_key_value.assign(key_states, value_states, layer_idx)

        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class PhiModel:
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RebelDynamicCache] = None,
        batch_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        forward_dict: Optional[Dict[str, classmethod]] = None,
        rotary_pos_emb=None,
    ) -> BaseModelOutputWithPast:
        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # embed positions
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        attention_mask = (1 - attention_mask) * torch.finfo(torch.float16).min

        # get cos,sin vector
        cos, sin = rotary_pos_emb(inputs_embeds, attention_mask.shape[-1])
        cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = forward_dict["model"](
                decoder_layer,
                hidden_states,
                layer_idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                batch_ids=batch_ids,
                cos=cos,
                sin=sin,
                rotary_pos_emb=rotary_pos_emb,
                forward_dict=forward_dict,
            )

            hidden_states = layer_outputs[0]

            updated_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # convert RebelDynamicCache to legacy Tuple[Tuple[torch.Tensor]]
        next_cache = updated_cache.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
