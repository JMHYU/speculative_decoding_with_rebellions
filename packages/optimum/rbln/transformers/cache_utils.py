from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


class RebelDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, current_steps) -> None:
        super().__init__()
        self.current_steps = current_steps

    def assign(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int, 
    ) -> None:
        self.key_cache[layer_idx] = key_states.squeeze(2)
        self.value_cache[layer_idx] = value_states.squeeze(2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        batch_idx: int,
        read_first_step: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx` and the batch 'batch_inx'
        based on self.current_step,
        """
        current_step = self.current_steps[0 if read_first_step else batch_idx]
        kend = current_step + key_states.shape[-2]
        vend = current_step + value_states.shape[-2]
        update_key_states = (
            self.key_cache[layer_idx][batch_idx]
            .unsqueeze(0)
            .unsqueeze(2)
            .slice_scatter(key_states, dim=-2, start=current_step, end=kend)
        )
        update_value_states = (
            self.value_cache[layer_idx][batch_idx]
            .unsqueeze(0)
            .unsqueeze(2)
            .slice_scatter(value_states, dim=-2, start=current_step, end=vend)
        )

        return update_key_states, update_value_states

    @classmethod
    def from_input_format(cls, position_ids, num_hidden_layer, *past_key_values) -> "DynamicCache":
        """Converts a cache in the rbln cache format (list of past_kv) into an equivalent `DynamicCache`."""

        batch, _ = position_ids.shape
        current_steps = [position_ids[b][0] for b in range(batch)]

        assert len(current_steps) == batch
        cache = cls(current_steps)

        for layer_idx in range(num_hidden_layer):
            key_states = past_key_values[layer_idx * 2]
            value_states = past_key_values[layer_idx * 2 + 1]
            cache.key_cache.append(key_states)
            cache.value_cache.append(value_states)

        return cache


class RebelDynamicCache_4D(RebelDynamicCache):
    def assign(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
    ) -> None:
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        batch_idx: int,
        read_first_step: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `keys` and `values` for the layer `layer_idx` and the batch 'batch_inx'
        based on self.current_step,
        """
        current_step = self.current_steps[0 if read_first_step else batch_idx]
        kend = current_step + keys.shape[-2]
        vend = current_step + values.shape[-2]
        update_keys = (
            self.key_cache[layer_idx][batch_idx].unsqueeze(0).slice_scatter(keys, dim=-2, start=current_step, end=kend)
        )
        update_values = (
            self.value_cache[layer_idx][batch_idx]
            .unsqueeze(0)
            .slice_scatter(values, dim=-2, start=current_step, end=vend)
        )

        return update_keys, update_values
