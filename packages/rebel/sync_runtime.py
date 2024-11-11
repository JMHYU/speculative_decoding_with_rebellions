# pylint: disable=import-outside-toplevel

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
import os
import typing
from ctypes import c_void_p

import numpy as np

from .core.runtime import CPU_NUM_THREADS
from .runtime_base import RuntimeBase

# To prevent import torch, here we use a trick.
TORCH = None
torch_empty = None  # pylint: disable=invalid-name
if typing.TYPE_CHECKING:
    import torch


class Runtime(RuntimeBase):
    """
    A Runtime object for executing a compiled neural network on an NPU.
    """

    def __init__(
        self,
        path: str,
        *,
        device: typing.Union[typing.List[int], int] = 0,
        input_info_index: typing.Optional[int] = None,
        tensor_type: typing.Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes a Runtime object for executing a compiled neural network on an NPU.

        Args:
            path (str): The path to the compiled rbln neural network file (*.rbln).
            device (int or List[int], optional): The device ID of the NPU to use for execution.
                Defaults to 0.
            tensor_type (str, optional): The object type of the tensor used in the
                `run` function. Possible values are:

                - "np": Uses np.ndarray type.
                - "pt": Uses torch.Tensor type.

                Defaults to "np".
        """
        super().__init__(
            path,
            device=device,
            non_blocking_mode=False,
            input_info_index=input_info_index,
            tensor_type=tensor_type,
            **kwargs,
        )
        self._set_input = self._handle[self._symbol_name + "_set_input"]
        self._set_output = self._handle[self._symbol_name + "_set_output"]

        if tensor_type == "pt":
            import torch

            global TORCH, torch_empty
            TORCH = torch
            torch_empty = torch.empty

    def run_pt_dynamo(
        self, *input_args: "torch.Tensor", **input_kwargs: "torch.Tensor"
    ) -> typing.Union["torch.Tensor", typing.List["torch.Tensor"]]:
        if self.cpu_threads is not None and isinstance(self.cpu_threads, int):
            os.environ[CPU_NUM_THREADS] = str(self.cpu_threads)
        # Return
        ret = _run_pt(self, *input_args, out=None, **input_kwargs)
        os.environ.pop(CPU_NUM_THREADS, None)
        return ret

    def run(
        self,
        *input_args: typing.Union[np.ndarray, "torch.Tensor"],
        out: typing.Optional[typing.List[typing.Union[np.ndarray, "torch.Tensor"]]] = None,
        **input_kwargs: typing.Union[np.ndarray, "torch.Tensor"],
    ) -> typing.Union[
        np.ndarray, "torch.Tensor", typing.List[typing.Union[np.ndarray, "torch.Tensor"]]
    ]:
        """Runs the compiled neural network with the given input tensors.

        Args:
            *input_args: Variable length argument list of input tensors.
                Each argument should be either a np.ndarray or a torch.Tensor.
            out: An optional list or tensor to store the output tensors.
                - If provided, it must contain pre-allocated tensors
                    with shapes matching the network's output shapes.
                - If not provided or set to `None`, new tensors
                    will be allocated to store the outputs.
            **input_kwargs: Arbitrary keyword arguments of input tensors.
                Each argument should be either a np.ndarray or a torch.Tensor.

        Returns:
            The output tensor(s) of the neural network. The return depends on the network's
                architecture and can be either a single tensor or a list of tensors.
                The tensor type (numpy.ndarray or torch.Tensor) is determined by the tensor_type
                provided during the Runtime object's initialization.
        """
        if self._tensor_type == "pt":
            return run_pt(self, *input_args, out=out, **input_kwargs)
        else:
            return run_np(self, *input_args, out=out, **input_kwargs)

    def forward(
        self,
        *input_args: typing.Union[np.ndarray, "torch.Tensor"],
        out: typing.Optional[typing.List[typing.Union[np.ndarray, "torch.Tensor"]]] = None,
        **input_kwargs: typing.Union[np.ndarray, "torch.Tensor"],
    ) -> typing.Union[
        np.ndarray, "torch.Tensor", typing.List[typing.Union[np.ndarray, "torch.Tensor"]]
    ]:
        """An alias for the `run` method.

        This method is provided for compatibility with PyTorch's naming convention.

        Args:
            *input_args: Variable length argument list of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.
            out: An optional list or tensor to store the output tensors.
                - If provided, it must contain pre-allocated tensors
                    with shapes matching the network's output shapes.
                - If not provided or set to `None`, new tensors
                    will be allocated to store the outputs.
            **input_kwargs: Arbitrary keyword arguments of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.

        Returns:
            The output tensor(s) of the neural network, as returned by the `run` method.
        """
        return self.run(*input_args, out=out, **input_kwargs)

    def __call__(
        self,
        *input_args: typing.Union[np.ndarray, "torch.Tensor"],
        out: typing.Optional[typing.List[typing.Union[np.ndarray, "torch.Tensor"]]] = None,
        **input_kwargs: typing.Union[np.ndarray, "torch.Tensor"],
    ) -> typing.Union[
        np.ndarray, "torch.Tensor", typing.List[typing.Union[np.ndarray, "torch.Tensor"]]
    ]:
        """Allows the Runtime object to be called as a function.

        This method is provided for convenience and compatibility with common neural network
            frameworks.

        Args:
            *input_args: Variable length argument list of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.
            out: An optional list or tensor to store the output tensors.
                - If provided, it must contain pre-allocated tensors
                    with shapes matching the network's output shapes.
                - If not provided or set to `None`, new tensors
                    will be allocated to store the outputs.
            **input_kwargs: Arbitrary keyword arguments of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.

        Returns:
            The output tensor(s) of the neural network, as returned by the `run` method.
        """
    
        return self.run(*input_args, out=out, **input_kwargs)


def _run_pt(
    runtime: "Runtime",
    *input_args: "torch.Tensor",
    out: typing.Optional[typing.List["torch.Tensor"]],
    **input_kwargs: "torch.Tensor",
) -> typing.Union["torch.Tensor", typing.List["torch.Tensor"]]:
    # Convert single tensor 'out' to a list
    if TORCH.is_tensor(out):
        out = [out]

    # Check for user given out
    runtime.raise_if_invalid_output_pytorch(out)

    inputs = runtime.prepare_inputs(*input_args, **input_kwargs)

    # Copy input
    for input_index, input_data in enumerate(inputs):
        runtime._set_input(input_index, c_void_p(input_data.data_ptr()))

    # Prepare output buffers
    outputs = []
    for output_index in range(runtime._num_outputs):
        # import pdb;pdb.set_trace()
        if out is not None:
            output = out[output_index]
        else:
            output = torch_empty(
                size=runtime._output_profile[output_index].shape,
                dtype=runtime._output_profile[output_index].dtype,
                device="cpu",
            )
        runtime._set_output(output_index, c_void_p(output.data_ptr()))
        outputs.append(output)

    # Run
    runtime._run()

    return outputs


def run_pt(
    runtime: "Runtime",
    *input_args: "torch.Tensor",
    out: typing.Optional[typing.List["torch.Tensor"]],
    **input_kwargs: "torch.Tensor",
) -> typing.Union["torch.Tensor", typing.List["torch.Tensor"]]:
    outputs = _run_pt(runtime, *input_args, out=out, **input_kwargs)
    # Return
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def run_np(
    runtime: "Runtime",
    *input_args: np.ndarray,
    out: typing.Optional[typing.List[np.ndarray]],
    **input_kwargs: np.ndarray,
) -> typing.Union[np.ndarray, typing.List[np.ndarray]]:
    # Convert single np.ndarray 'out' to a list
    if isinstance(out, np.ndarray):
        out = [out]

    # check for user given out
    runtime.raise_if_invalid_outputs(out)

    inputs = runtime.prepare_inputs(*input_args, **input_kwargs)

    # Copy input
    for input_index, input_data in enumerate(inputs):
        runtime._set_input(input_index, c_void_p(input_data.ctypes.data))

    # Prepare output buffers
    outputs = []
    for output_index in range(runtime._num_outputs):
        if out is not None:
            output = out[output_index]
        else:
            output = np.empty(
                shape=runtime._output_profile[output_index].shape,
                dtype=runtime._output_profile[output_index].dtype,
            )
        runtime._set_output(output_index, c_void_p(output.ctypes.data))
        outputs.append(output)

    # Run
    runtime._run()

    # Return
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs
