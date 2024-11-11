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
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from optimum.exporters import TasksManager
from transformers import AutoConfig, AutoModel

from ....modeling_base import RBLNModel
from ....modeling_config import RBLNConfig
from ...models.controlnet import RBLNControlNetModel


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RBLNMultiControlNetModel(RBLNModel):
    def __init__(
        self,
        models: List[RBLNControlNetModel],
        **kwargs,
    ):
        self.nets = models
        self.dtype = torch.float32

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        def get_model_from_task(
            task: str,
            model_name_or_path: Union[str, Path],
            **kwargs,
        ):
            return MultiControlNetModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path, **kwargs)

        tasktmp = TasksManager.get_model_from_task
        configtmp = AutoConfig.from_pretrained
        modeltmp = AutoModel.from_pretrained
        TasksManager.get_model_from_task = get_model_from_task
        AutoConfig.from_pretrained = ControlNetModel.load_config
        AutoModel.from_pretrained = MultiControlNetModel.from_pretrained
        rt = super().from_pretrained(*args, **kwargs)
        AutoConfig.from_pretrained = configtmp
        AutoModel.from_pretrained = modeltmp
        TasksManager.get_model_from_task = tasktmp
        return rt

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        **kwargs,
    ) -> RBLNModel:
        idx = 0
        controlnets = []
        model_path_to_load = model_id

        while os.path.isdir(model_path_to_load):
            controlnet = RBLNControlNetModel.from_pretrained(model_path_to_load, export=False, **kwargs)
            controlnets.append(controlnet)
            rbln_config = RBLNConfig.load(model_path_to_load)
            idx += 1
            model_path_to_load = model_id + f"_{idx}"

        return cls(
            controlnets,
            rbln_config=rbln_config,
            **kwargs,
        )

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        for idx, model in enumerate(self.nets):
            suffix = "" if idx == 0 else f"_{idx}"
            real_save_path = save_directory + suffix
            model.save_pretrained(real_save_path)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ):
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            output = controlnet.model[0](
                sample=sample.contiguous(),
                timestep=timestep.float(),
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                conditioning_scale=torch.tensor(scale),
            )

            down_samples, mid_sample = output[:-1], output[-1]

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample
