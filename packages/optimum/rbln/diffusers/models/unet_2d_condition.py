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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from optimum.exporters import TasksManager
from transformers import AutoConfig, AutoModel, PretrainedConfig

from ...modeling_base import RBLNModel
from ...modeling_config import RBLNCompileConfig, RBLNConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


class _UNet_SD(torch.nn.Module):
    def __init__(self, unet: "UNet2DConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        *down_and_mid_block_additional_residuals: Optional[Tuple[torch.Tensor]],
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        if len(down_and_mid_block_additional_residuals) != 0:
            down_block_additional_residuals, mid_block_additional_residual = (
                down_and_mid_block_additional_residuals[:-1],
                down_and_mid_block_additional_residuals[-1],
            )
        else:
            down_block_additional_residuals, mid_block_additional_residual = None, None

        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return unet_out


class _UNet_SDXL(torch.nn.Module):
    def __init__(self, unet: "UNet2DConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        *down_and_mid_block_additional_residuals: Optional[Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        if len(down_and_mid_block_additional_residuals) == 2:
            added_cond_kwargs = {
                "text_embeds": down_and_mid_block_additional_residuals[0],
                "time_ids": down_and_mid_block_additional_residuals[1],
            }
            down_block_additional_residuals = None
            mid_block_additional_residual = None
        elif len(down_and_mid_block_additional_residuals) > 2:
            added_cond_kwargs = {
                "text_embeds": down_and_mid_block_additional_residuals[-2],
                "time_ids": down_and_mid_block_additional_residuals[-1],
            }
            down_block_additional_residuals, mid_block_additional_residual = (
                down_and_mid_block_additional_residuals[:-3],
                down_and_mid_block_additional_residuals[-3],
            )
        else:
            added_cond_kwargs = {}
            down_block_additional_residuals = None
            mid_block_additional_residual = None

        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return unet_out


class RBLNUNet2DConditionModel(RBLNModel):
    model_type = "rbln_model"
    auto_model_class = AutoModel  # feature extraction

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.in_features = self.rbln_config.model_cfg.get("in_features", None)
        if self.in_features is not None:

            @dataclass
            class LINEAR1:
                in_features: int

            @dataclass
            class ADDEMBEDDING:
                linear_1: LINEAR1

            self.add_embedding = ADDEMBEDDING(LINEAR1(self.in_features))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        def get_model_from_task(
            task: str,
            model_name_or_path: Union[str, Path],
            **kwargs,
        ):
            return UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path, **kwargs)

        tasktmp = TasksManager.get_model_from_task
        configtmp = AutoConfig.from_pretrained
        modeltmp = AutoModel.from_pretrained
        TasksManager.get_model_from_task = get_model_from_task
        if kwargs.get("export", None):
            # This is an ad-hoc to workaround save null values of the config.
            # if export, pure optimum(not optimum-rbln) loads config using AutoConfig
            # and diffusers model do not support loading by AutoConfig.
            AutoConfig.from_pretrained = lambda *args, **kwargs: None
        else:
            AutoConfig.from_pretrained = UNet2DConditionModel.load_config
        AutoModel.from_pretrained = UNet2DConditionModel.from_pretrained
        rt = super().from_pretrained(*args, **kwargs)
        AutoConfig.from_pretrained = configtmp
        AutoModel.from_pretrained = modeltmp
        TasksManager.get_model_from_task = tasktmp
        return rt

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        if model.config.addition_embed_type == "text_time":
            return _UNet_SDXL(model).eval()
        else:
            return _UNet_SD(model).eval()

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_text_model_hidden_size = rbln_kwargs.get("text_model_hidden_size", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_in_features = rbln_kwargs.get("in_features", None)
        rbln_use_encode = rbln_kwargs.get("use_encode", None)
        rbln_img_width = rbln_kwargs.get("img_width", None)
        rbln_img_height = rbln_kwargs.get("img_height", None)
        rbln_vae_scale_factor = rbln_kwargs.get("vae_scale_factor", None)
        rbln_is_controlnet = rbln_kwargs.get("is_controlnet", None)

        if rbln_max_seq_len is None:
            rbln_max_seq_len = 77
        if rbln_batch_size is None:
            rbln_batch_size = 1

        if rbln_use_encode:
            if rbln_img_width is None or rbln_img_height is None or rbln_vae_scale_factor is None:
                raise ValueError(
                    "rbln_img_width, rbln_img_height, and rbln_vae_scale_factor must be provided when rbln_use_encode is True"
                )
            input_width = rbln_img_width // rbln_vae_scale_factor
            input_height = rbln_img_height // rbln_vae_scale_factor
        else:
            input_width, input_height = model_config.sample_size, model_config.sample_size

        input_info = [
            (
                "sample",
                [
                    rbln_batch_size,
                    model_config.in_channels,
                    input_height,
                    input_width,
                ],
                "float32",
            ),
            ("timestep", [], "float32"),
            (
                "encoder_hidden_states",
                [
                    rbln_batch_size,
                    rbln_max_seq_len,
                    model_config.cross_attention_dim,
                ],
                "float32",
            ),
        ]

        if rbln_is_controlnet:
            if len(model_config.block_out_channels) > 0:
                input_info.extend(
                    [
                        (
                            f"down_block_additional_residuals_{i}",
                            [rbln_batch_size, model_config.block_out_channels[0], input_height, input_width],
                            "float32",
                        )
                        for i in range(3)
                    ]
                )
            if len(model_config.block_out_channels) > 1:
                input_info.append(
                    (
                        "down_block_additional_residuals_3",
                        [rbln_batch_size, model_config.block_out_channels[0], input_height // 2, input_width // 2],
                        "float32",
                    )
                )
                input_info.extend(
                    [
                        (
                            f"down_block_additional_residuals_{i}",
                            [rbln_batch_size, model_config.block_out_channels[1], input_height // 2, input_width // 2],
                            "float32",
                        )
                        for i in range(4, 6)
                    ]
                )
            if len(model_config.block_out_channels) > 2:
                input_info.append(
                    (
                        f"down_block_additional_residuals_{6}",
                        [rbln_batch_size, model_config.block_out_channels[1], input_height // 4, input_width // 4],
                        "float32",
                    )
                )
                input_info.extend(
                    [
                        (
                            f"down_block_additional_residuals_{i}",
                            [rbln_batch_size, model_config.block_out_channels[2], input_height // 4, input_width // 4],
                            "float32",
                        )
                        for i in range(7, 9)
                    ]
                )
            if len(model_config.block_out_channels) > 3:
                input_info.extend(
                    [
                        (
                            f"down_block_additional_residuals_{i}",
                            [rbln_batch_size, model_config.block_out_channels[3], input_height // 8, input_width // 8],
                            "float32",
                        )
                        for i in range(9, 12)
                    ]
                )
            input_info.append(
                (
                    "mid_block_additional_residual",
                    [
                        rbln_batch_size,
                        model_config.block_out_channels[-1],
                        input_height // 2 ** (len(model_config.block_out_channels) - 1),
                        input_width // 2 ** (len(model_config.block_out_channels) - 1),
                    ],
                    "float32",
                )
            )

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        if hasattr(model_config, "addition_embed_type") and model_config.addition_embed_type == "text_time":
            if rbln_text_model_hidden_size is None:
                rbln_text_model_hidden_size = 768
            if rbln_in_features is None:
                rbln_in_features = model_config.projection_class_embeddings_input_dim
            rbln_compile_config.input_info.append(
                ("text_embeds", [rbln_batch_size, rbln_text_model_hidden_size], "float32")
            )
            rbln_compile_config.input_info.append(("time_ids", [rbln_batch_size, 6], "float32"))

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update(
            {
                "max_seq_len": rbln_max_seq_len,
                "batch_size": rbln_batch_size,
                "use_encode": rbln_use_encode,
            }
        )

        if rbln_in_features is not None:
            rbln_config.model_cfg["in_features"] = rbln_in_features

        return rbln_config

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = {},
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        arg order : latent_model_input, t, prompt_embeds
        """
        added_cond_kwargs = {} if added_cond_kwargs is None else added_cond_kwargs

        if down_block_additional_residuals is not None:
            down_block_additional_residuals = [t.contiguous() for t in down_block_additional_residuals]
            return (
                super().forward(
                    sample.contiguous(),
                    timestep.float(),
                    encoder_hidden_states,
                    *down_block_additional_residuals,
                    mid_block_additional_residual,
                    **added_cond_kwargs,
                ),
            )

        return (
            super().forward(
                sample.contiguous(),
                timestep.float(),
                encoder_hidden_states,
                **added_cond_kwargs,
            ),
        )
