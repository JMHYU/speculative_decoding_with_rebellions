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

import importlib

from transformers import AutoConfig


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def get_rbln_cls(
        cls,
        model_id,
        *args,
        **kwargs,
    ):
        # kwargs.update({"return_unused_kwargs": True})
        config = AutoConfig.from_pretrained(model_id, return_unused_kwargs=True, **kwargs)[0]

        if len(config.architectures) > 1:
            raise ValueError(
                f"Model with ID '{model_id}' has multiple architectures defined in the configuration: "
                f"{config.architectures}. `_BaseAutoModelClass` require exactly one architecture. "
            )

        architecture_name = config.architectures[0]
        if architecture_name not in cls._model_mapping.values():
            raise ValueError(
                f"The 'RBLN{architecture_name}' architecture is not supported by `{cls.__name__}.from_pretrained()`."
                "Please use the appropriate class's `from_pretrained()` method to load this model."
            )

        rbln_class_name = "RBLN" + architecture_name
        module = importlib.import_module("optimum.rbln")

        try:
            rbln_cls = getattr(module, rbln_class_name)
        except AttributeError as e:
            raise AttributeError(
                f"Class '{rbln_class_name}' not found in 'optimum.rbln' module for model ID '{model_id}'. "
                "Ensure that the class name is correctly mapped and available in the 'optimum.rbln' module."
            ) from e

        return rbln_cls

    @classmethod
    def from_pretrained(
        cls,
        model_id,
        *args,
        **kwargs,
    ):
        rbln_cls = cls.get_rbln_cls(model_id, *args, **kwargs)
        return rbln_cls.from_pretrained(model_id, *args, **kwargs)
