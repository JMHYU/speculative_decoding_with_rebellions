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

from .core.compiled_model_core import CompiledModelCore
from .core.exception import RBLNRuntimeError

if typing.TYPE_CHECKING:
    from .async_runtime import AsyncRuntime
    from .sync_runtime import Runtime


class RBLNCompiledModel(CompiledModelCore):
    """
    A class representing a compiled model.

    This class provides an interface for creating and managing runtimes for the compiled model.

    Example usage:
    ```python
    # Load a compiled model
    model = RBLNCompiledModel("/path/to/compiled/model.rbln")

    # Create a runtime for execution on NPU 0
    runtime = model.create_runtime(device=0)

    # Get the total device memory allocation
    total_alloc = model.get_total_device_alloc()

    # Get the device memory allocation per NPU
    alloc_per_npu = model.get_alloc_per_node()
    ```
    """

    def create_runtime(
        self,
        name: typing.Optional[str] = None,
        *,
        input_info_index: typing.Optional[int] = None,
        tensor_type: typing.Optional[str] = None,
        device: typing.Optional[int] = None,
        **kwargs,
    ) -> "Runtime":
        """
        Create runtime with this binaries.
        Note that this function is exclusive to `create_async_runtime`.
        Once you create a runtime by `create_runtime` with the instance,
        you can't call `create_async_runtime`.

        Args:
            device (int, optional): The device ID of the NPU to use for execution. Defaults to 0.
            tensor_type (str, optional): The object type of the tensor used in the
                `run` function. Possible values are:

                - "np": Uses np.ndarray type.
                - "pt": Uses torch.Tensor type.

                Defaults to "np".

        Returns:
            Runtime object that can be run on the RBLN ATOM
        """
        if self._runtime_mode == "async":
            raise RBLNRuntimeError(
                "Async type of runtime has already been created for this instance"
            )
        self._runtime_mode = "sync"
        return self._create_runtime(
            name,
            input_info_index=input_info_index,
            non_blocking_mode=False,
            tensor_type=tensor_type,
            device=device,
            **kwargs,
        )

    def create_async_runtime(
        self,
        name: typing.Optional[str] = None,
        *,
        input_info_index: typing.Optional[int] = None,
        tensor_type: typing.Optional[str] = None,
        device: typing.Optional[int] = None,
        **kwargs,
    ) -> "AsyncRuntime":
        """
        Create asynchronous version of runtime with this binaries.
        Note that this function is exclusive to `create_runtime`.
        Once you create an asynchronous runtime by `create_async_runtime` with the instance,
        you can't call `create_runtime`.

        Args:
            device (int, optional): The device ID of the NPU to use for execution. Defaults to 0.
            tensor_type (str, optional): The object type of the tensor used in the
                `run` function. Possible values are:

                - "np": Uses np.ndarray type.
                - "pt": Uses torch.Tensor type.

                Defaults to "np".

        Returns:
            Asynchronous runtime object that can be run on the RBLN ATOM
        """
        if self._runtime_mode == "sync":
            raise RBLNRuntimeError(
                "Sync type of runtime has already been created for this instance"
            )
        self._runtime_mode = "async"
        return self._create_runtime(
            name,
            input_info_index=input_info_index,
            non_blocking_mode=True,
            tensor_type=tensor_type,
            device=device,
            **kwargs,
        )

    def get_total_device_alloc(self, non_blocking_mode=False) -> int:
        """
        Retrieves the total device memory allocation (in bytes) required for the compiled graph
        across all NPUs.

        Args:
            non_blocking_mode (bool, optional): If True, the returned allocation size accounts
                for additional buffer requirements when operating in non-blocking (asynchronous)
                mode. Defaults to False.

        Returns:
            int: The total device memory allocation (in bytes) required for the compiled graph.
        """
        return sum(self._get_alloc_per_node(non_blocking_mode))

    def get_alloc_per_node(self, non_blocking_mode=False) -> typing.List[int]:
        """
        Retrieves the device memory allocation (in bytes) required for the compiled graph on
        each individual NPU.

        Args:
            non_blocking_mode (bool, optional): If True, the returned allocation sizes account
                for additional buffer requirements when operating in non-blocking (asynchronous)
                mode. Defaults to False.

        Returns:
            List[int]: A list containing the device memory allocation (in bytes) for each NPU.
                The length of the list corresponds to the number of NPUs used for
                tensor parallelism.
        """
        return self._get_alloc_per_node(non_blocking_mode)
