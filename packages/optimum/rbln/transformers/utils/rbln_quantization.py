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


from typing import Any

import torch
from torch.nn import Linear, Parameter
from torch.nn import functional as F


# Constants
QUANTIZED_WEIGHTS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def update_layers_to_quantized(module: torch.nn.Module) -> None:
    """
    Updates specified linear layers to quantized (qlinear) layers in the given module.
    """
    processed_layers = []

    for name, layer in module.named_modules():
        if is_target_for_qlinear_replacement(name, layer):
            parent_module, layer_name = get_parent_and_child(module, name)
            setattr(parent_module, layer_name, create_qlinear(layer))
            processed_layers.append(name)

    if processed_layers:
        print(f"Updated the following linear layers to quantized layers:\n {{{', '.join(processed_layers)}}}")


def is_target_for_qlinear_replacement(layer_name: str, layer: torch.nn.Module) -> bool:
    """
    Checks if a layer is a target for qlinear replacement.
    """
    return layer_name.split(".")[-1] in QUANTIZED_WEIGHTS and isinstance(layer, torch.nn.Linear)


def get_parent_and_child(module: torch.nn.Module, full_name: str) -> tuple:
    """
    Splits the full layer name to retrieve the parent module and the child layer.
    """
    *parent_address, child_name = full_name.split(".")
    parent_module = access_attribute(module, parent_address)
    return parent_module, child_name


def access_attribute(obj: Any, attributes: list[str]) -> Any:
    """
    Recursively accesses a nested attribute from an object using a list of attribute names.
    """
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def create_qlinear(layer: Linear) -> Linear:
    """
    Converts a standard linear layer to a quantized linear (qlinear) layer with a custom forward pass.
    """
    def qlinear_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != self.scales.dtype:
            raise TypeError(f"Expected input dtype {self.scales.dtype}, but got {inputs.dtype}")

        w_fp = self.weight.type(inputs.dtype)
        w_fp *= self.scales.view(-1, 1)
        return F.linear(inputs, w_fp, self.bias)

    # Convert weight to int8 and add scale parameter
    layer.weight = Parameter(layer.weight.to(torch.int8), requires_grad=False)
    layer.scales = Parameter(torch.ones(layer.out_features, dtype=torch.float32), requires_grad=False)
    layer.forward = lambda inputs: qlinear_forward(layer, inputs)

    return layer
