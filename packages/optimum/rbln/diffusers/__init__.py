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

from typing import TYPE_CHECKING

from diffusers.pipelines.pipeline_utils import ALL_IMPORTABLE_CLASSES, LOADABLE_CLASSES
from transformers.utils import _LazyModule


LOADABLE_CLASSES["optimum.rbln"] = {"RBLNBaseModel": ["save_pretrained", "from_pretrained"]}
ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES["optimum.rbln"])


_import_structure = {
    "pipelines": [
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNMultiControlNetModel",
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
    ],
    "models": ["RBLNAutoencoderKL", "RBLNUNet2DConditionModel", "RBLNControlNetModel"],
}

if TYPE_CHECKING:
    from .models import RBLNAutoencoderKL, RBLNControlNetModel, RBLNUNet2DConditionModel
    from .pipelines import (
        RBLNMultiControlNetModel,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionPipeline,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
        RBLNStableDiffusionXLImg2ImgPipeline,
        RBLNStableDiffusionXLPipeline,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
