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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    AutoModelForVision2Seq,
    LlavaNextForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast

from ....modeling_base import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyOutput


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class LoopVisionTower:
    def __init__(self, vision_tower: RBLNModel) -> None:
        self.vision_tower = vision_tower

    def forward(self, *args, **kwargs):
        # Loop instead of batch
        # shape of pixel_values : [batch, num_patches, num_channel, height, width]
        pixel_values = args[0]

        batch_size = pixel_values.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.vision_tower.model[0](pixel_values[i : i + 1]))

        last_hidden_states = [output[0] for output in outputs]
        pooler_output = [output[1] for output in outputs]

        # FIXME:: This can be optimized using out= API of rbln runtime.
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        pooler_output = torch.cat(pooler_output, dim=0)

        hidden_states = [output[2:] for output in outputs]  # batch x (hidden x 1)

        hidden_states = tuple(
            torch.cat(tuple((hidden_states[n][i] for n in range(batch_size))), dim=0)
            for i in range(len(hidden_states[0]))
        )  # hidden x (batch,)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.vision_tower)


class LoopProjector:
    def __init__(self, multi_modal_projector) -> None:
        self.multi_modal_projector = multi_modal_projector

    def forward(self, *args, **kwargs):
        # Loop instead of batch
        image_feature = args[0]

        batch_size = image_feature.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.multi_modal_projector(image_feature[i : i + 1]))

        # FIXME:: This can be optimized using out= API of rbln runtime.
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.vision_tower)


class RBLNLlavaNextForConditionalGeneration(RBLNModel):
    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(LlavaNextForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "LlavaNextForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        save_dict = {}
        save_dict["image_newline"] = model.image_newline
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = self.rbln_submodules[1]
        self.multi_modal_projector = LoopProjector(self.model[0])

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.image_newline = artifacts["image_newline"]

        # Copied from the original class
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        return super().__post_init__(**kwargs)

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        *args,
        rbln_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        # Optimum's TasksManager does not handle Llava.
        kwargs = cls.update_kwargs(kwargs)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id, *args, **kwargs)
        return model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        return model.multi_modal_projector

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs={},
    ) -> RBLNConfig:
        vision_feature_select_strategy = rbln_kwargs.get("vision_feature_select_strategy", None)

        # 1. Multi-modal projection layer
        batch_size = rbln_kwargs.get("rbln_batch_size", None)
        if batch_size is None:
            batch_size = 1

        feature_size = model_config.vision_config.hidden_size

        # See forward function to see more details.
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else model_config.vision_feature_select_strategy
        )

        # Calculating `num_positions` : See CLIPVisionEmbeddings of transformers for more details.
        num_positions = (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2 + 1
        if vision_feature_select_strategy == "default":
            selected_image_feature_dim = num_positions - 1
        else:
            selected_image_feature_dim = num_positions

        input_info = [("image_features", [batch_size, selected_image_feature_dim, feature_size], "float32")]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(rbln_cls=cls.__name__, compile_cfgs=[rbln_compile_config], rbln_kwargs=rbln_kwargs)
        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        past_cached_length=None,
        **kwargs,
    ):
        # Prepare HF generation
        is_prefill_phase = past_cached_length is None
        batch_size = input_ids.shape[0]

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_cached_length=past_cached_length,  # Not affect
            attention_mask=attention_mask,
            **kwargs,
        )

        if is_prefill_phase:
            model_inputs["past_cached_length"] = torch.zeros((batch_size, 1), dtype=torch.int32)
        else:
            model_inputs["past_cached_length"] = past_cached_length + 1

        model_inputs.update(
            {
                # "position_ids": position_ids or cache_positions,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
                "attention_mask": attention_mask,
            }
        )
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

    def _merge_vllm_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        mask = input_ids == placeholder_token_id
        num_expected_tokens = mask.sum().item()
        assert isinstance(num_expected_tokens, int)

        if multimodal_embeddings.shape[0] != num_expected_tokens:
            raise ValueError(
                f"Attempted to assign {inputs_embeds[mask].shape} = {multimodal_embeddings.shape} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            )

        inputs_embeds[mask] = multimodal_embeddings
        return inputs_embeds

    def _embed(
        self,
        input_ids: torch.LongTensor,
        image_sizes: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int,
        vision_feature_select_strategy: str,
        cache_position: torch.Tensor,
        past_cached_length: torch.Tensor,
        from_vllm_prefill: bool = False,
    ) -> List[torch.Tensor]:
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # 1. Extract the input embeddings
        # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
        for_inputs_embeds_ids = input_ids.clone()
        for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0

        inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
            # ! infer image_num_patches from image_sizes
            image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=self.config.image_grid_pinpoints,
                    patch_size=self.config.vision_config.image_size,
                )
                for imsize in image_sizes
            ]
            # figure out if pixel_values is concatenated or stacked
            if pixel_values.dim() == 5:
                # stacking when input is (batch_size, num_patches, num_channels, height, width)
                _pixel_values_list = [
                    pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                ]
                pixel_values = torch.cat(_pixel_values_list, dim=0)
            elif pixel_values.dim() != 4:
                # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

            image_features = self.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_features.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature

            image_features = self.multi_modal_projector(selected_image_feature)
            image_features = torch.split(image_features, image_num_patches, dim=0)

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
            )

            inputs_embeds = inputs_embeds.to(image_features.dtype)

            if from_vllm_prefill:
                self._merge_vllm_multimodal_embeddings(
                    input_ids, inputs_embeds, image_features, self.config.image_token_index
                )
            else:
                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                )

            cache_position = torch.arange(0, inputs_embeds.shape[1], dtype=torch.int32).unsqueeze_(0)

        # pixel_values is not None but is empty ---> text only cases
        elif (
            pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0 or pixel_values is None
        ):
            pass

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif pixel_values is not None and input_ids.shape[1] == 1 and past_cached_length is not None:
            cache_position = past_cached_length

        return inputs_embeds, cache_position

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Union[List[torch.Tensor], torch.Tensor] = None,  # vllm keyword argument
        batch_idx: Optional[int] = None,
        past_cached_length: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        from_vllm_prefill = isinstance(cache_position, torch.Tensor) and cache_position.shape[-1] > 1
        from_hf_generate_prefill = isinstance(input_ids, list)

        if inputs_embeds is not None:
            raise NotImplementedError("Specifying inputs_embeds is not supported.")

        if from_hf_generate_prefill:
            inputs_embeds = []
            batch_size = len(input_ids)

            # Get the number of images in the prompt
            special_image_token_masks = [input_id == self.config.image_token_index for input_id in input_ids]
            num_special_image_tokens = [torch.sum(mask, dim=-1) for mask in special_image_token_masks]

            # Split images for each prompt
            pixel_values = pixel_values.split(num_special_image_tokens, dim=0)
            image_sizes = image_sizes.split(num_special_image_tokens, dim=0)

            for b_idx in range(batch_size):
                embed, cache_pos = self._embed(
                    input_ids=input_ids[b_idx],
                    image_sizes=image_sizes[b_idx] if image_sizes is not None else None,
                    attention_mask=torch.ones_like(input_ids[b_idx]),
                    pixel_values=pixel_values[b_idx] if pixel_values is not None else None,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    cache_position=cache_position[b_idx],
                    past_cached_length=past_cached_length[b_idx : b_idx + 1],
                )
                inputs_embeds.append(embed)
                cache_position[b_idx] = cache_pos
                past_cached_length[b_idx] += embed.shape[1]

        elif from_vllm_prefill:
            inputs_embeds, cache_position = self._embed(
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                cache_position=cache_position,
                past_cached_length=past_cached_length,
                from_vllm_prefill=from_vllm_prefill,
            )
        else:
            # Decoding step
            inputs_embeds, cache_position = self._embed(
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                cache_position=cache_position,
                past_cached_length=past_cached_length,
            )

        outputs: RBLNDecoderOnlyOutput = self.language_model(
            inputs_embeds=inputs_embeds,
            batch_idx=batch_idx,
            cache_position=cache_position,
            past_cached_length=past_cached_length,
        )

        return outputs

    # Almost copied from : https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/llava_next/modeling_llava_next.py
    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens


# Almost copied from : https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/llava_next/modeling_llava_next.py
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


# Almost copied from : https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/llava_next/modeling_llava_next.py
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


# Almost copied from : https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/llava_next/modeling_llava_next.py
def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


# Almost copied from : https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/llava_next/modeling_llava_next.py
def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`Union[torch.LongTensor, np.ndarray, Tuple[int, int]):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches