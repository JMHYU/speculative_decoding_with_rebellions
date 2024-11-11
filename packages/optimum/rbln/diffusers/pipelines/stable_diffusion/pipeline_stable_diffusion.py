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
"""RBLNStableDiffusionPipeline class for inference of diffusion models on rbln devices."""

from diffusers import StableDiffusionPipeline

from ....modeling_base import RBLNBaseModel
from ....transformers import RBLNCLIPTextModel
from ....utils.runtime_utils import ContextRblnConfig
from ...models import RBLNAutoencoderKL, RBLNUNet2DConditionModel


class RBLNStableDiffusionPipeline(StableDiffusionPipeline):
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        """
        Pipeline for text-to-image generation using Stable Diffusion.

        This model inherits from [`StableDiffusionPipeline`]. Check the superclass documentation for the generic methods
        implemented for all pipelines (downloading, saving, running on a particular device, etc.).

        It implements the methods to convert a pre-trained Stable Diffusion pipeline into a RBLNStableDiffusion pipeline by:
        - transferring the checkpoint weights of the original into an optimized RBLN graph,
        - compiling the resulting graph using the RBLN compiler.

        Args:
            model_id (`Union[str, Path]`):
                Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing a model saved using [`~OptimizedModel.save_pretrained`],
        """
        export = kwargs.pop("export", None)
        model_save_dir = kwargs.pop("model_save_dir", None)
        rbln_config = kwargs.pop("rbln_config", None)
        rbln_kwargs, _ = RBLNBaseModel.resolve_rbln_config(rbln_config, kwargs)

        device = rbln_kwargs.get("device", None)
        device_map = rbln_kwargs.get("device_map", None)
        create_runtimes = rbln_kwargs.get("create_runtimes", None)
        optimize_host_memory = rbln_kwargs.get("optimize_host_memory", None)

        with ContextRblnConfig(
            device=device,
            device_map=device_map,
            create_runtimes=create_runtimes,
            optimze_host_mem=optimize_host_memory,
        ):
            model = super().from_pretrained(pretrained_model_name_or_path=model_id, **kwargs)

        if export is None or export is False:
            return model

        do_classifier_free_guidance = (
            rbln_kwargs.pop("guidance_scale", 5.0) > 1.0 and model.unet.config.time_cond_proj_dim is None
        )

        vae = RBLNAutoencoderKL.from_pretrained(
            model_id=model_id,
            subfolder="vae",
            export=True,
            model_save_dir=model_save_dir,
            rbln_unet_sample_size=model.unet.config.sample_size,
            rbln_use_encode=False,
            rbln_config={**rbln_kwargs},
        )
        text_encoder = RBLNCLIPTextModel.from_pretrained(
            model_id=model_id,
            subfolder="text_encoder",
            export=True,
            model_save_dir=model_save_dir,
            rbln_config={**rbln_kwargs},
        )

        batch_size = rbln_kwargs.pop("batch_size", 1)
        unet_batch_size = batch_size * 2 if do_classifier_free_guidance else batch_size

        unet = RBLNUNet2DConditionModel.from_pretrained(
            model_id=model_id,
            subfolder="unet",
            export=True,
            model_save_dir=model_save_dir,
            rbln_max_seq_len=text_encoder.config.max_position_embeddings,
            rbln_batch_size=unet_batch_size,
            rbln_use_encode=False,
            rbln_is_controlnet=True if "controlnet" in model.config.keys() else False,
            rbln_config={**rbln_kwargs},
        )

        if model_save_dir is not None:
            # To skip saving original pytorch modules
            del (model.vae, model.text_encoder, model.unet)

            # Direct calling of `save_pretrained` causes config.unet = (None, None).
            # So config must be saved again, later.
            model.save_pretrained(model_save_dir)

        # replace modules
        model.vae = vae
        model.text_encoder = text_encoder
        model.unet = unet

        # update config to be able to load from file.
        update_dict = {
            "vae": ("optimum.rbln", "RBLNAutoencoderKL"),
            "text_encoder": ("optimum.rbln", "RBLNCLIPTextModel"),
            "unet": ("optimum.rbln", "RBLNUNet2DConditionModel"),
        }
        model.register_to_config(**update_dict)

        if model_save_dir is not None:
            # overwrite to replace incorrect config
            model.save_config(model_save_dir)

        model.models = [vae.model[0], text_encoder.model[0], unet.model[0]]

        if optimize_host_memory is False:
            model.compiled_models = [vae.compiled_models[0], text_encoder.compiled_models[0], unet.compiled_models[0]]

        return model
