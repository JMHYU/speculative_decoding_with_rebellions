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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from diffusers import ControlNetModel
from optimum.exporters import TasksManager
from transformers import AutoConfig, AutoModel, PretrainedConfig

from ...modeling_base import RBLNModel
from ...modeling_config import RBLNCompileConfig, RBLNConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


logger = logging.getLogger(__name__)


class _ControlNetModel(torch.nn.Module):
    def __init__(self, controlnet: "ControlNetModel"):
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale,
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ):
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=None,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample


class _ControlNetModel_Cross_Attention(torch.nn.Module):
    def __init__(self, controlnet: "ControlNetModel"):
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale,
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ):
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample


class RBLNControlNetModel(RBLNModel):
    model_type = "rbln_model"
    auto_model_class = AutoModel  # feature extraction

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.use_encoder_hidden_states = any(
            item[0] == "encoder_hidden_states" for item in self.rbln_config.compile_cfgs[0].input_info
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if "subfolder" in kwargs:
            del kwargs["subfolder"]

        def get_model_from_task(
            task: str,
            model_name_or_path: Union[str, Path],
            **kwargs,
        ):
            return ControlNetModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path, **kwargs)

        tasktmp = TasksManager.get_model_from_task
        configtmp = AutoConfig.from_pretrained
        modeltmp = AutoModel.from_pretrained
        TasksManager.get_model_from_task = get_model_from_task
        AutoConfig.from_pretrained = ControlNetModel.load_config
        AutoModel.from_pretrained = ControlNetModel.from_pretrained
        rt = super().from_pretrained(*args, **kwargs)
        AutoConfig.from_pretrained = configtmp
        AutoModel.from_pretrained = modeltmp
        TasksManager.get_model_from_task = tasktmp
        return rt

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        use_encoder_hidden_states = False
        for down_block in model.down_blocks:
            if use_encoder_hidden_states := getattr(down_block, "has_cross_attention", False):
                break

        if use_encoder_hidden_states:
            return _ControlNetModel_Cross_Attention(model).eval()
        else:
            return _ControlNetModel(model).eval()

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
        rbln_img_width = rbln_kwargs.get("img_width", None)
        rbln_img_height = rbln_kwargs.get("img_height", None)
        rbln_vae_scale_factor = rbln_kwargs.get("vae_scale_factor", None)

        if rbln_batch_size is None:
            rbln_batch_size = 1

        if rbln_max_seq_len is None:
            rbln_max_seq_len = 77

        if rbln_img_width is None or rbln_img_height is None or rbln_vae_scale_factor is None:
            raise ValueError("rbln_img_width, rbln_img_height, and rbln_vae_scale_factor must be provided")

        input_width = rbln_img_width // rbln_vae_scale_factor
        input_height = rbln_img_height // rbln_vae_scale_factor

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
        ]

        use_encoder_hidden_states = any(element != "DownBlock2D" for element in model_config.down_block_types)
        if use_encoder_hidden_states:
            input_info.append(
                (
                    "encoder_hidden_states",
                    [
                        rbln_batch_size,
                        rbln_max_seq_len,
                        model_config.cross_attention_dim,
                    ],
                    "float32",
                )
            )

        input_info.append(("controlnet_cond", [rbln_batch_size, 3, rbln_img_height, rbln_img_width], "float32"))
        input_info.append(("conditioning_scale", [], "float32"))

        if hasattr(model_config, "addition_embed_type") and model_config.addition_embed_type == "text_time":
            if rbln_text_model_hidden_size is None:
                rbln_text_model_hidden_size = 768
            input_info.append(("text_embeds", [rbln_batch_size, rbln_text_model_hidden_size], "float32"))
            input_info.append(("time_ids", [rbln_batch_size, 6], "float32"))

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update(
            {
                "max_seq_len": rbln_max_seq_len,
                "batch_size": rbln_batch_size,
                "img_width": rbln_img_width,
                "img_height": rbln_img_height,
                "vae_scale_factor": rbln_vae_scale_factor,
            }
        )

        return rbln_config

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: torch.Tensor = 1.0,
        added_cond_kwargs: Dict[str, torch.Tensor] = {},
        **kwargs,
    ):
        """
        The [`ControlNetModel`] forward method.
        """
        added_cond_kwargs = {} if added_cond_kwargs is None else added_cond_kwargs
        if self.use_encoder_hidden_states:
            output = super().forward(
                sample.contiguous(),
                timestep.float(),
                encoder_hidden_states,
                controlnet_cond,
                torch.tensor(conditioning_scale),
                **added_cond_kwargs,
            )
        else:
            output = super().forward(
                sample.contiguous(),
                timestep.float(),
                controlnet_cond,
                torch.tensor(conditioning_scale),
                **added_cond_kwargs,
            )
        down_block_res_samples = output[:-1]
        mid_block_res_sample = output[-1]

        return down_block_res_samples, mid_block_res_sample
