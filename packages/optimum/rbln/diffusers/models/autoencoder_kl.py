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
from typing import TYPE_CHECKING, Any, Dict, List, Union

import rebel
import torch  # noqa: I001
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from optimum.exporters import TasksManager
from transformers import AutoConfig, AutoModel, PretrainedConfig

from ...modeling_base import RBLNModel
from ...modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ...utils.runtime_utils import RBLNPytorchRuntime


if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig

logger = logging.getLogger(__name__)


class RBLNRuntimeVAEEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        moments = self.forward(x.contiguous())
        posterior = DiagonalGaussianDistribution(moments)
        return AutoencoderKLOutput(latent_dist=posterior)


class RBLNRuntimeVAEDecoder(RBLNPytorchRuntime):
    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return (self.forward(z),)


class RBLNAutoencoderKL(RBLNModel):
    model_type = "rbln_model"
    config_name = "config.json"
    auto_model_class = AutoModel  # feature extraction

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        self.rbln_use_encode = self.rbln_config.model_cfg["use_encode"]
        if self.rbln_use_encode:
            self.encoder = RBLNRuntimeVAEEncoder(runtime=self.model[0], main_input_name="x")
            self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[1], main_input_name="z")
        else:
            self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[0], main_input_name="z")

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        def compile_img2img():
            encoder_model = _VAEEncoder(model)
            decoder_model = _VAEDecoder(model)
            encoder_model.eval()
            decoder_model.eval()

            enc_compiled_model = cls.compile(encoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
            dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[1])

            return {"encoder": enc_compiled_model, "decoder": dec_compiled_model}

        def compile_text2img():
            decoder_model = _VAEDecoder(model)
            decoder_model.eval()

            dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])

            return dec_compiled_model

        if rbln_config.model_cfg.get("use_encode", False):
            return compile_img2img()
        else:
            return compile_text2img()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        def get_model_from_task(
            task: str,
            model_name_or_path: Union[str, Path],
            **kwargs,
        ):
            return AutoencoderKL.from_pretrained(pretrained_model_name_or_path=model_name_or_path, **kwargs)

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
            AutoConfig.from_pretrained = AutoencoderKL.load_config

        AutoModel.from_pretrained = AutoencoderKL.from_pretrained
        rt = super().from_pretrained(*args, **kwargs)
        AutoConfig.from_pretrained = configtmp
        AutoModel.from_pretrained = modeltmp
        TasksManager.get_model_from_task = tasktmp
        return rt

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_unet_sample_size = rbln_kwargs.get("unet_sample_size", None)
        rbln_img_width = rbln_kwargs.get("img_width", None)
        rbln_img_height = rbln_kwargs.get("img_height", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_use_encode = rbln_kwargs.get("use_encode", None)
        rbln_vae_scale_factor = rbln_kwargs.get("vae_scale_factor", None)

        if rbln_batch_size is None:
            rbln_batch_size = 1

        model_cfg = {}

        if rbln_use_encode:
            model_cfg["img_width"] = rbln_img_width
            model_cfg["img_height"] = rbln_img_height

            vae_enc_input_info = [
                ("x", [rbln_batch_size, model_config.in_channels, rbln_img_height, rbln_img_width], "float32")
            ]
            vae_dec_input_info = [
                (
                    "z",
                    [
                        rbln_batch_size,
                        model_config.latent_channels,
                        rbln_img_height // rbln_vae_scale_factor,
                        rbln_img_width // rbln_vae_scale_factor,
                    ],
                    "float32",
                )
            ]

            enc_rbln_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info)
            dec_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=vae_dec_input_info)

            compile_cfgs = [enc_rbln_compile_config, dec_rbln_compile_config]
            rbln_config = RBLNConfig(
                rbln_cls=cls.__name__,
                compile_cfgs=compile_cfgs,
                rbln_kwargs=rbln_kwargs,
            )
            rbln_config.model_cfg.update(model_cfg)
            return rbln_config

        if rbln_unet_sample_size is None:
            rbln_unet_sample_size = 64

        model_cfg["unet_sample_size"] = rbln_unet_sample_size
        vae_config = RBLNCompileConfig(
            input_info=[
                (
                    "z",
                    [
                        rbln_batch_size,
                        model_config.latent_channels,
                        rbln_unet_sample_size,
                        rbln_unet_sample_size,
                    ],
                    "float32",
                )
            ],
        )
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[vae_config],
            rbln_kwargs=rbln_kwargs,
        )
        rbln_config.model_cfg.update(model_cfg)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls, compiled_models: List[rebel.RBLNCompiledModel], rbln_device_map: Dict[str, int]
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
            return [compiled_models[0].create_runtime(tensor_type="pt", device=device_val)]

        device_vals = [rbln_device_map["encoder"], rbln_device_map["decoder"]]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device_val)
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        posterior = self.encoder.encode(x)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.decoder.decode(z)


class _VAEDecoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKL"):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        vae_out = self.vae.decode(z, return_dict=False)
        return vae_out


class _VAEEncoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKL"):
        super().__init__()
        self.vae = vae

    def encode(self, x: torch.FloatTensor, return_dict: bool = True):
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        return moments

    def forward(self, x):
        vae_out = _VAEEncoder.encode(self.vae, x, return_dict=False)
        return vae_out
