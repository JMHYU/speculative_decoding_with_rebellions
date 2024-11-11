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

import asyncio
import typing
from ctypes import c_void_p

import numpy as np

from .core._env import ENV
from .runtime_base import RuntimeBase

torch_empty = None  # pylint: disable=invalid-name
if typing.TYPE_CHECKING:
    import torch


def run_pt(
    runtime: "AsyncRuntime", *input_args: "torch.Tensor", **input_kwargs: "torch.Tensor"
) -> typing.Union["torch.Tensor", typing.List["torch.Tensor"]]:
    inputs = runtime.prepare_inputs(*input_args, **input_kwargs)

    # Reserve output memory buffer
    outputs: typing.Union[torch.Tensor, typing.List[torch.Tensor]] = []
    for output_index in range(runtime._num_outputs):
        output = torch_empty(
            runtime._output_profile[output_index].shape,
            dtype=runtime._output_profile[output_index].dtype,
            device="cpu",
        )
        outputs.append(output)

    # Call run function
    rid: int = runtime._run(
        *[c_void_p(v.data_ptr()) for v in inputs],
        *[c_void_p(v.data_ptr()) for v in outputs],
    )

    # Return async task
    return AsyncTask(runtime, rid, outputs[0] if len(outputs) == 1 else outputs, inputs)


def run_np(runtime: "AsyncRuntime", *input_args: np.ndarray, **input_kwargs: np.ndarray):
    inputs = runtime.prepare_inputs(*input_args, **input_kwargs)

    # Reserve output memory buffer
    outputs: typing.Union[np.ndarray, typing.List[np.ndarray]] = []
    for output_index in range(runtime._num_outputs):
        output = np.empty(
            runtime._output_profile[output_index].shape,
            runtime._output_profile[output_index].dtype,
        )
        outputs.append(output)

    # Call run function
    rid: int = runtime._run(
        *[c_void_p(v.ctypes.data) for v in inputs],
        *[c_void_p(v.ctypes.data) for v in outputs],
    )

    # Return async task
    return AsyncTask(runtime, rid, outputs[0] if len(outputs) == 1 else outputs, inputs)


class AsyncTask:
    def __init__(
        self,
        runtime: "AsyncRuntime",
        rid: int,
        outputs: typing.Union[np.ndarray, "torch.Tensor"],
        inputs: typing.Union[np.ndarray, "torch.Tensor"],
    ):
        self.runtime = runtime
        self.rid = rid
        self.outputs = outputs

        # to hold the memory of input data, we keep the reference of it.
        # this may increase memory usage.
        self.inputs = inputs

    def wait(
        self, timeout: typing.Optional[float] = None
    ) -> typing.Union[
        np.ndarray, "torch.Tensor", typing.List[typing.Union[np.ndarray, "torch.Tensor"]]
    ]:
        """Waits for the asynchronous task to complete and returns the result.

        This method blocks the calling thread until the task is completed or the specified timeout
            is reached.

        Args:
            timeout (typing.Optional[float], optional): The maximum amount of time (in seconds) to
                wait for the task to complete. If None, the method will wait indefinitely until the
                task is completed. Defaults to None.

        Returns:
            The output tensor(s) of the neural network. The return depends on the network's
                architecture and can be either a single tensor or a list of tensors. The tensor type
                (numpy.ndarray or torch.Tensor) is determined by the tensor_type provided during the
                AsyncRuntime object's initialization.
        """
        if self.outputs is None:
            raise RuntimeError("Cannot wait for an async task that has already returned.")

        if timeout is None or timeout < 0:
            timeout = 0.0

        if ENV == "DEV":
            print(f"awaiting ({self.rid})...")
        self.runtime._await(self.rid, int(timeout * 1000))
        if ENV == "DEV":
            print(f"await done ({self.rid}).")

        output = self.outputs
        self.outputs = None
        self.inputs = None
        return output


class AsyncRuntime(RuntimeBase):
    """An AsyncRuntime object for executing a compiled neural network asynchronously on an NPU."""

    def __init__(
        self,
        path: str,
        *,
        device: int = 0,
        input_info_index: typing.Optional[int] = None,
        tensor_type: typing.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes an AsyncRuntime object for executing a compiled neural network asynchronously
        on an NPU.

        Args:
            path (str): The path to the compiled rbln neural network file (*.rbln).
            device (int, optional): The device ID of the NPU to use for execution. Defaults to 0.
            tensor_type (str, optional): The object type of the tensor used in the
                `run` function. Possible values are:

                - "np": Uses np.ndarray type.
                - "pt": Uses torch.Tensor type.

                Defaults to "np".
        """
        super().__init__(
            path,
            device=device,
            non_blocking_mode=True,
            input_info_index=input_info_index,
            tensor_type=tensor_type,
            **kwargs,
        )

        self._await = self._handle[self._symbol_name + "_await"]

        if tensor_type == "pt":
            import torch

            global torch_empty
            torch_empty = torch.empty

    def run(
        self,
        *input_args: typing.Union[np.ndarray, "torch.Tensor"],
        **input_kwargs: typing.Union[np.ndarray, "torch.Tensor"],
    ) -> AsyncTask:
        """Runs the compiled neural network asynchronously with the given input tensors.

        Args:
            *input_args: Variable length argument list of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.
            **input_kwargs: Arbitrary keyword arguments of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.

        Returns:
            AsyncTask: An asynchronous task object representing the neural network execution.
                The task object can be used to wait for the neural network execution to finish.
        """
        if self._tensor_type == "pt":
            return run_pt(self, *input_args, **input_kwargs)
        else:
            return run_np(self, *input_args, **input_kwargs)

    async def async_run(
        self,
        *input_args: typing.Union[np.ndarray, "torch.Tensor"],
        **input_kwargs: typing.Union[np.ndarray, "torch.Tensor"],
    ) -> typing.Union[
        np.ndarray, "torch.Tensor", typing.List[typing.Union[np.ndarray, "torch.Tensor"]]
    ]:
        """Runs the compiled neural network asynchronously and returns the result awaitably.

        This method is a coroutine that can be used with the `await` keyword to asynchronously run
        the neural network and retrieve the result.

        Args:
            *input_args: Variable length argument list of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.
            **input_kwargs: Arbitrary keyword arguments of input tensors.
                Each argument should be either a numpy.ndarray or a torch.Tensor.

        Returns:
            The output tensor(s) of the neural network. The return type depends on the network's
                architecture and can be either a single tensor or a list of tensors. The tensor type
                (numpy.ndarray or torch.Tensor) is determined by the tensor_type provided during the
                AsyncRuntime object's initialization.
        """
        task = self.run(*input_args, **input_kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, task.wait)
