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

import numpy as np
from prettytable import PrettyTable

from .core.runtime import RuntimeCore
from .core.utility import sizeof_fmt

if typing.TYPE_CHECKING:
    import torch

# To prevent import torch, here we use a trick.
torch_tensor = None  # pylint: disable=invalid-name


class RuntimeBase(RuntimeCore):
    def __init__(
        self,
        path: str,
        *,
        device: typing.Union[int, typing.List[int]] = 0,
        non_blocking_mode: bool = False,
        tensor_type: typing.Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes a RuntimeBase object for executing a compiled neural network on an NPU.

        This is the base class for runtime execution.

        Args:
            path (str): The path to the compiled rbln neural network file (*.rbln).
            device (int or List[int], optional): The device ID of the NPU(s) to use for execution.
                Can be a single integer or a list of integers for multi-device execution.
                Defaults to 0.
            non_blocking_mode (bool, optional): If True, enables non-blocking (async) execution.
                Defaults to False.
            tensor_type (str, optional): The object type of the tensor used in the
                `run` function. Possible values are:

                - "np": Uses np.ndarray type.
                - "pt": Uses torch.Tensor type.

                Defaults to "np".

        """
        if tensor_type is None:
            tensor_type = "np"
        if tensor_type == "pt":
            import torch  # pylint: disable=import-outside-toplevel

            global torch_tensor
            torch_tensor = torch.Tensor

        super().__init__(
            path,
            device=device,
            non_blocking_mode=non_blocking_mode,
            tensor_type=tensor_type,
            **kwargs,
        )

    def raise_if_invalid_outputs(self, out: typing.Optional[typing.List[np.ndarray]]):
        if out is None:
            # Ignore if out is None.
            return

        if not isinstance(out, (list, tuple)):
            raise TypeError(
                "Invalid type for 'out': expected a list or tuple, "
                f"but received {type(out).__name__}. "
            )

        if len(out) != len(self._output_profile):
            raise RuntimeError(
                f"Mismatch in the number of output tensors: expected {len(self._output_profile)}, "
                f"but received {len(out)}. Ensure that the 'out' parameter contains the correct "
                "number of tensors corresponding to the model's output profile."
            )

        for i, output_ in enumerate(out):
            if output_ is None:
                raise ValueError(f"The output({i}) is not specified.")
            if not isinstance(output_, np.ndarray):
                raise RuntimeError(
                    f"The input({i}) must be a numpy array type, " f"not ({type(output_)})."
                )
            if not output_.data.c_contiguous:
                raise RuntimeError(f"The output({i}) must be contiguous.")

            target_dtype = self._output_profile[i].dtype
            source_dtype = output_.dtype

            if source_dtype != target_dtype:
                raise TypeError(
                    f"The output({i}) (dtype={source_dtype}) has a dtype "
                    f"different to required dtype ({target_dtype})."
                )

            target_shape = tuple(self._output_profile[i].shape)
            source_shape = output_.shape

            if target_shape != source_shape:
                raise TypeError(
                    f"The output({i}) (shape={source_shape}) has a shape "
                    f"different to required shape {target_shape}."
                )

    def raise_if_invalid_output_pytorch(self, out: typing.Optional[typing.List["torch.Tensor"]]):
        if out is None:
            # Ignore if out is None.
            return

        if not isinstance(out, (list, tuple)):
            raise TypeError(
                "Invalid type for 'out': expected a list or tuple, "
                f"but received {type(out).__name__}. "
            )

        if len(out) != len(self._output_profile):
            raise RuntimeError(
                f"Mismatch in the number of output tensors: expected {len(self._output_profile)}, "
                f"but received {len(out)}. Ensure that the 'out' parameter contains the correct "
                "number of tensors corresponding to the model's output profile."
            )

        for i, output_ in enumerate(out):
            if output_ is None:
                raise ValueError(f"The output({i}) is not specified.")

            if not isinstance(output_, torch_tensor):
                raise RuntimeError(
                    f"The output({i}) must be a torch.Tensor type, not " f"({type(output_)})."
                )
            if not output_.is_contiguous():
                raise RuntimeError(f"The output({i}) must be contiguous.")
            if output_.get_device() != -1:
                raise RuntimeError(f"The output({i}) must be on the cpu.")

            target_dtype = self._output_profile[i].dtype
            source_dtype = output_.dtype

            if source_dtype != target_dtype:
                raise TypeError(
                    f"The output({i}) (dtype={source_dtype}) has a dtype "
                    f"different to required dtype ({target_dtype})."
                )

            target_shape = self._output_profile[i].shape
            source_shape = output_.shape

            if target_shape != source_shape:
                raise TypeError(
                    f"The output({i}) (shape={tuple(source_shape)}) has a shape "
                    f"different to required shape {target_shape}."
                )

    def raise_if_invalid_inputs(self, inputs: typing.List[np.ndarray]) -> bool:
        for i, input_ in enumerate(inputs):
            if input_ is None:
                raise ValueError(f"The input({self._index_to_input_name[i]}) is not specified.")
            if not isinstance(input_, np.ndarray):
                raise RuntimeError(
                    f"The input({self._index_to_input_name[i]}) must be a numpy array type, "
                    f"not ({type(input_)})."
                )
            if not input_.data.c_contiguous:
                raise RuntimeError(f"The input({self._index_to_input_name[i]}) must be contiguous.")

            target_dtype = self._input_profile[i].dtype
            source_dtype = input_.dtype

            if source_dtype != target_dtype:
                raise TypeError(
                    f"{self._index_to_input_name[i]} (dtype={source_dtype}) has a dtype "
                    f"different to required dtype ({target_dtype})."
                )

            target_shape = tuple(self._input_profile[i].shape)
            source_shape = input_.shape

            if target_shape != source_shape:
                raise TypeError(
                    f"{self._index_to_input_name[i]} (shape={source_shape}) has a shape "
                    f"different to required shape {target_shape}."
                )

    def raise_if_invalid_inputs_pytorch(self, inputs: typing.List["torch.Tensor"]) -> bool:
        for i, input_ in enumerate(inputs):
            if input_ is None:
                raise ValueError(f"The input({self._index_to_input_name[i]}) is not specified.")

            if not isinstance(input_, torch_tensor):
                raise RuntimeError(
                    f"The input({self._index_to_input_name[i]}) must be a torch.Tensor type, not "
                    f"({type(input_)})."
                )
            if not input_.is_contiguous():
                raise RuntimeError(f"The input({self._index_to_input_name[i]}) must be contiguous.")
            if input_.get_device() != -1:
                raise RuntimeError(f"The input({self._index_to_input_name[i]}) must be on the cpu.")

            target_dtype = self._input_profile[i].dtype
            source_dtype = input_.dtype

            if source_dtype != target_dtype:
                raise TypeError(
                    f"{self._index_to_input_name[i]} (dtype={source_dtype}) has a dtype "
                    f"different to required dtype ({target_dtype})."
                )

            target_shape = self._input_profile[i].shape
            source_shape = input_.shape

            if target_shape != source_shape:
                raise TypeError(
                    f"{self._index_to_input_name[i]} (shape={tuple(source_shape)}) has a shape "
                    f"different to required shape {target_shape}."
                )

        return True

    def prepare_inputs(
        self,
        *input_args: np.ndarray,
        **input_kwargs: np.ndarray,
    ) -> typing.List[np.ndarray]:
        # Merge input parameters
        inputs: typing.List[np.ndarray] = [None] * self._num_inputs
        if len(input_args) + len(input_kwargs) > self._num_inputs:
            raise RuntimeError(
                f"Number of inputs ({len(input_args) + len(input_kwargs)}) exceeds"
                f" the expected number ({self._num_inputs})."
            )

        for i, v in enumerate(input_args):
            inputs[i] = v
        for k, v in input_kwargs.items():
            if k not in self._input_name_to_index:
                raise RuntimeError(
                    f"Unknown key [{k}]. Possible keys : {list(self._input_name_to_index.keys())}"
                )
            inputs[self._input_name_to_index[k]] = v

        if self._tensor_type == "pt":
            self.raise_if_invalid_inputs_pytorch(inputs)
        else:
            self.raise_if_invalid_inputs(inputs)

        return inputs

    def model_description(self):
        table = PrettyTable(["I/O Type", "Key", "Shape", "Dtype"], align="l")
        table.title = "Compiled Model Description"

        for i in range(len(self._input_profile)):
            key = self._index_to_input_name[i] if i in self._index_to_input_name else i
            row = ["Input", key, self._input_profile[i].shape, self._input_profile[i].dtype]
            table.add_row(row, divider=(i == len(self._input_profile) - 1))

        for i in range(len(self._output_profile)):
            row = ["Output", i, self._output_profile[i].shape, self._output_profile[i].dtype]
            table.add_row(row)

        table_str = table.get_string()
        width_table = len(table_str.split("\n")[0])
        meta_table = PrettyTable(["Label", "Quantity"], header=False)
        meta_table._min_width = {"Quantity": width_table - 7 - len("Compiler version")}
        version = str(self._meta.get("compiler_ver", "DEV"))
        meta_table.add_row(["Compiler version", version], divider=True)
        npu = str(self._meta.get("npu", "Unknown"))
        meta_table.add_row(["NPU", npu], divider=True)
        for device_id, alloc in sorted(
            zip(self.user_devices, self.alloc_per_device), key=lambda x: x[0]
        ):
            is_mda = len(self.user_devices) > 1
            device_id_str = f"[device={device_id}]" if is_mda else ""
            meta_table.add_row([f"Memory{device_id_str}", sizeof_fmt(alloc)])

        return str(table) + "\n" + str(meta_table)

    def __repr__(self):
        return self.desc
