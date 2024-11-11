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
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)

from ...cache_utils import RebelDynamicCache


class DecoderOnlyWrapper(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.config = model.config
        self.model = model.model
        self.lm_head = model.lm_head

        self.head_dim = (
            self.config.head_dim
            if hasattr(self.config, "head_dim")
            else self.config.hidden_size // self.config.num_attention_heads
        )
        self.max_position_embeddings = (
            self.config.max_position_embeddings if max_seq_len > self.config.max_position_embeddings else max_seq_len
        )
        self.max_seq_len = max_seq_len
        self.rope_scaling = getattr(self.config, "rope_scaling", None)
        self.rotary_emb = self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "linear":
                rotary_emb = LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.config.rope_theta,
                    max_seq_len=self.max_seq_len,
                )
            elif scaling_type == "dynamic":
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.config.rope_theta,
                    max_seq_len=self.max_seq_len,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

        return rotary_emb

    def get_forward_dict(self):
        forward_dict = {
            "wrapper": DecoderOnlyModel.forward,
            "model": DecoderOnlyDecoderLayer.forward,
            "decoder_layer": DecoderOnlyAttention.forward,
        }
        return forward_dict

    def forward(
        self,
        input_ids_or_inputs_embeds,
        attention_mask,
        cache_position,
        batch_position,
        query_idx,
        *past_key_values,
    ):
        # [AIHA-yjh] RBLNDecoderOnlyModelForCausalLM.get_compiled_model에서 넘겨준 dummy inputs가 넘어온다.
        if input_ids_or_inputs_embeds.shape[1] == 1:
            rbln_batch_position = None
        else:
            rbln_batch_position = batch_position

        if input_ids_or_inputs_embeds.ndim == 2:
            # input_ids
            input_ids = input_ids_or_inputs_embeds
            inputs_embeds = None
        elif input_ids_or_inputs_embeds.ndim == 3:
            # inputs_embeds
            input_ids = None
            inputs_embeds = input_ids_or_inputs_embeds
        else:
            raise NotImplementedError(f"Unknown ndim of input : {input_ids_or_inputs_embeds.ndim}")

        # Formatting list of past_kv to DynamicCache class.
        past_key_values = RebelDynamicCache.from_input_format(
            cache_position,
            self.config.num_hidden_layers,
            *past_key_values,
        )

        forward_dict = self.get_forward_dict()
        outputs = forward_dict["wrapper"](
            self.model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=cache_position,
            past_key_values=past_key_values,
            batch_ids=rbln_batch_position,
            rotary_pos_emb=self.rotary_emb,
            forward_dict=forward_dict,
        )


        # last_hidden_state
        hidden_states = outputs[0]

        if batch_position >= 0:
            hidden_states = hidden_states[:, query_idx].unsqueeze(1)

        logits = self.lm_head(hidden_states)

        output = (logits,) + outputs[1:]

        # 리턴값을 추가함
        return output, batch_position + query_idx, hidden_states


class DecoderOnlyAttention:
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[RebelDynamicCache] = None,
        batch_index: Optional[int] = None,
        output_attentions: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

                # reshape for removing repeat_kv
                attn_weight = torch.matmul(query_state, key_state.transpose(3, 4)) / math.sqrt(self.head_dim)

                attn_weight = attn_weight + attn_mask

                # upcast attention to fp32
                attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weight, value_state)

                # reshape for removing repeat_kv
                attn_output = attn_output.view(1, self.num_heads, q_len, self.head_dim)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(1, q_len, self.num_heads * self.head_dim)

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

            attn_weight = torch.matmul(query_states, key_states.transpose(3, 4)) / math.sqrt(self.head_dim)
            attn_weight = attn_weight + attention_mask

            # upcast attention to fp32
            attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weight, value_states)

            # reshape for removing repeat_kv
            attn_output = attn_output.view(1, self.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weight = None

        return attn_output, attn_weight, key_states, value_states


class DecoderOnlyDecoderLayer:
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
        forward_dict: Optional[Dict[str, classmethod]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weight, k, v = forward_dict["decoder_layer"](
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
            **kwargs,
        )
        past_key_value.assign(k, v, layer_idx)

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weight,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class DecoderOnlyModel:
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
                forward_dict=forward_dict,
            )

            hidden_states = layer_outputs[0]

            updated_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        hidden_states = self.norm(hidden_states)

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


def slice_and_unsqueeze_cos_sin(cos, sin, position_ids, unsqueeze_dim=1):
    """Slice cos[position_ids], sin[position_ids] vector for the query."""
    if position_ids.shape[0] > 1:
        cos_all = []
        sin_all = []
        for i in range(position_ids.shape[0]):
            cos_all.append(cos[position_ids[i : i + 1]].unsqueeze(unsqueeze_dim))
            sin_all.append(sin[position_ids[i : i + 1]].unsqueeze(unsqueeze_dim))
        cos = torch.cat(cos_all, dim=0)
        sin = torch.cat(sin_all, dim=0)
    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        device = self.inv_freq.device

        positions_ids = torch.arange(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions_ids, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    def forward(self, x, seq_len):
        return (
            self._cos_cached[:seq_len].to(dtype=x.dtype),
            self._sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        max_seq_len=2048,
    ):
        super().__init__(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            scaling_factor=scaling_factor,
        )
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        if max_seq_len > max_position_embeddings:
            positions_ids = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            positions_ids = positions_ids / self.scaling_factor
            freqs = torch.outer(positions_ids, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

            self._cos_cached = torch.cat([self._cos_cached, cos[max_position_embeddings:]], dim=0)
            self._sin_cached = torch.cat([self._sin_cached, sin[max_position_embeddings:]], dim=0)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        max_seq_len=2048,
    ):
        super().__init__(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            scaling_factor=scaling_factor,
        )
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype
        if max_seq_len > max_position_embeddings:
            position_ids = torch.arange(max_position_embeddings, max_seq_len, dtype=dtype).view(-1, 1)
            seq_len = position_ids + 1
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))

            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

            freqs = position_ids * inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

            self._cos_cached = torch.cat([self._cos_cached, cos], dim=0)
            self._sin_cached = torch.cat([self._sin_cached, sin], dim=0)
