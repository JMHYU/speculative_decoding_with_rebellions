# pylint: disable=import-outside-toplevel,redefined-builtin

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

import typing
from collections import UserList
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    try:
        import numpy as np
        import tensorflow.python.framework.dtypes as tf_dtypes
        import torch
    except ImportError:
        pass

DType = typing.Union[str, "torch.dtype", "tf_dtypes.DType", "np.dtype", typing.Any]
TypeInputInfo = typing.List[typing.Tuple[str, typing.List[int], DType]]


class ArbitraryDType:
    """
    ArbitraryDType("float32").to("torch")
    ArbitraryDType(torch.float32)
    ArbitraryDType(torch.float32).to("tensorflow")
    """

    def __init__(self, type: DType) -> None:
        if isinstance(type, str):
            self.type: str = type
        else:
            self.type: str = repr(type).split(".")[-1]
            if self.type.endswith("'>"):  # numpy
                self.type = self.type[:-2]

    def to(self, framework: str) -> DType:  # pylint: disable=invalid-name
        # framework : ["torch", "tensorflow", "pt", "tf", "np"]
        if framework == "torch" or framework == "pt":
            import torch

            try:
                return getattr(torch, self.type)
            except AttributeError:
                raise AttributeError(f"Unknown dtype {self.type}")

        elif framework == "tensorflow" or framework == "tf":
            import tensorflow as tf

            try:
                return getattr(tf, self.type)
            except AttributeError:
                raise AttributeError(f"Unknown dtype {self.type}")
        elif framework == "numpy" or framework == "np":
            import numpy as np

            try:
                return getattr(np, self.type)
            except AttributeError:
                raise AttributeError(f"Unknown dtype {self.type}")
        elif framework == "str":
            return self.type
        else:
            raise ValueError(f'Unknown framework [{framework}]. ["torch", "tensorflow", "numpy"]')

    def __repr__(self) -> str:
        return self.type


@dataclass
class Node:
    name: str
    shape: typing.List[int]
    dtype: DType

    def __post_init__(self):
        self.dtype = ArbitraryDType(self.dtype)


class InputInfo(UserList):
    def convert_to_tuple(self, target):
        info = []
        if target == "tvm":
            # convert to (name, (shape, dtype)) format
            for node in self:
                info.append((node.name, (node.shape, node.dtype.to("str"))))
        elif target == "rbln":
            # convert to (name, shape, dtype) format
            for node in self:
                info.append((node.name, node.shape, node.dtype.to("str")))
        else:
            raise ValueError(f"Invalid InputInfo target : {target}")
        return info


class InputInfos(UserList):
    # input_infos (M x `input_info`)
    # - input_info (N x `node`)
    # --      node(node.name, node.shape, node.dtype)
    @staticmethod
    def from_input_info(input_info: typing.Union[typing.List[TypeInputInfo], TypeInputInfo]):
        def raise_if_name_is_keyword(name: str):
            if name == "out":
                raise RuntimeError(
                    "Oops, invalid input name given. "
                    "`out` is a reserved keyword. "
                    "Please compile with another input name."
                )

        def get_depth(info, depth=1):
            if isinstance(info, str):
                return depth
            elif isinstance(info, (list, tuple)):
                if len(info) > 0:
                    return get_depth(info[0], depth + 1)
                else:
                    raise ValueError(f"Invalid `input_info`, got {info}.")
            else:
                raise ValueError(f"Invalid `input_info`, got {info}.")

        depth = get_depth(input_info, 1)

        input_infos = InputInfos()

        if depth == 3:
            # [("x", [1,3,224,224], torch.float32), ("y", [1,64], torch.int64)]
            input_infos.append(InputInfo())
            for name, shape, dtype in input_info:
                raise_if_name_is_keyword(name)
                node = Node(name, shape, dtype)
                input_infos[0].append(node)
        elif depth == 4:
            # [
            #   [("x", [1,3,224,224], torch.float32), ("y", [1,64], torch.int64)],
            #   [("x", [8,3,512,512], torch.float32), ("y", [8,64], torch.int64)],
            # ]
            num_inputs = len(input_info[0])
            expected_node_names = [name for name, _, _ in input_info[0]]

            for _input_info in input_info:
                if len(_input_info) != num_inputs:
                    raise AttributeError(
                        "The number of inputs of each input_info should be same!"
                        "\n"
                        f"Expected {num_inputs} ({input_info[0]}), "
                        f"but got {len(_input_info)} ({_input_info})"
                    )

                node_names = [name for name, _, _ in _input_info]
                if expected_node_names != node_names:
                    raise AttributeError(
                        f"The names of inputs should be same at all input specs."
                        "\n"
                        f"Expected {expected_node_names}, Got {node_names}"
                    )

                input_infos.append(InputInfo())
                for name, shape, dtype in _input_info:
                    raise_if_name_is_keyword(name)
                    node = Node(name, shape, dtype)
                    input_infos[-1].append(node)
        else:
            raise ValueError(f"Invalid `input_info`, got {input_info}")

        return input_infos
