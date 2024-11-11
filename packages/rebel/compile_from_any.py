# pylint: disable=import-outside-toplevel, redefined-builtin

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

from .compiled_model import RBLNCompiledModel
from .core.compilation._impl import (
    TypeInputInfo,
    compile,
    tf_function_to_ir,
    tf_graph_def_to_ir,
    torch_to_ir,
    torchscript_to_ir,
)

if typing.TYPE_CHECKING:
    try:
        import tensorflow as tf
        import tensorflow.compat.v1 as tf_v1
        import torch
    except ImportError:
        pass

__all__ = [
    "compile_from_tf_graph_def",
    "compile_from_tf_function",
    "compile_from_torch",
    "compile_from_torchscript",
]


def compile_from_tf_graph_def(
    graph_def: "tf_v1.GraphDef",
    outputs: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    layout: str = "NHWC",
    **kwargs,
) -> RBLNCompiledModel:
    """
    Compile a model from TensorFlow `GraphDef`.
    This function allows you to compile TensorFlow V1.x legacy models.
    If you are using TensorFlow V2 as a default, we recommend compiling the model
    using `compile_from_tf_function` in its function form.

    Args:
        graph_def (tf.compat.v1.GraphDef): A tensorflow graph definition
            in the form of a protocol buffer
        outputs: A string or list of the name of output node(s) (Optional).
            If not specified, then the last node is assumed to be the graph output.
            This may be useful when the graph has multiple outputs.
        layout: Layout of the tensor used internally in the model. One of "NHWC" or "NCHW"
    Returns:
        compiled_model (RBLNCompiledModel): Compiled model that can be run on the RBLN NPU
    """  # pylint: disable=line-too-long  # noqa: E501

    mod = tf_graph_def_to_ir(graph_def, outputs, layout)
    compiled_model = compile(mod, **kwargs)
    return compiled_model


def compile_from_tf_function(
    func: "tf.types.experimental.GenericFunction",
    input_info: typing.Union[typing.List[TypeInputInfo], TypeInputInfo],
    outputs: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    layout: str = "NHWC",
    **kwargs,
) -> RBLNCompiledModel:
    """
    Compile a model from `tf.function`.
    Note that the input function should not be concretized by `get_concrete_function` method.

    Args:
        func (tf.types.experimental.GenericFunction): A tensorflow function
        input_info: A list of input information,
            with each information described in triple format (name, shape, dtype).
        outputs: A string or list of the name of output node(s) (Optional).
            If not specified, then the last node is assumed to be the graph output.
            This may be useful when the graph has multiple outputs.
        layout: Layout of the tensor used internally in the model. One of "NHWC" or "NCHW"
    Returns:
        compiled_model (RBLNCompiledModel): Compiled model that can be run on the RBLN NPU
    """
    mod = tf_function_to_ir(func, input_info, outputs, layout)
    compiled_model = compile(mod, **kwargs)
    return compiled_model


def compile_from_torch(
    mod: "torch.nn.Module",
    input_info: typing.Union[typing.List[TypeInputInfo], TypeInputInfo],
    **kwargs,
) -> RBLNCompiledModel:
    """
    Compile a model from `torch.nn.Module`.

    Args:
        mod: A pytorch function
        input_info: A list of input information,
            with each information described in a triple format (name, shape, dtype).

    Returns:
        compiled_model (RBLNCompiledModel): Compiled model that can be run on the RBLN NPU
    """
    mod = torch_to_ir(mod=mod, input_info=input_info)
    compiled_model = compile(mod, **kwargs)
    return compiled_model


def compile_from_torchscript(
    mod: "torch.jit.ScriptModule",
    input_names: typing.Optional[typing.List[typing.Optional[str]]] = None,
    input_info: typing.Optional[typing.Union[typing.List[TypeInputInfo], TypeInputInfo]] = None,
    **kwargs,
) -> RBLNCompiledModel:
    """
    Compile a model from `torch.jit.ScriptModule`, a result of `torch.jit.trace` function.

    Args:
        mod (torch.jit.ScriptModule): A pytorch jit-traced model
        input_names (legacy): A list of input names having same length as module's input
            If `name` is specified as 'None`, it'll derive name from `mod`,
            name of corresponding input in `forward` function.
        input_info: A list of input information,
            with each information described in a triple format (name, shape, dtype).
    Returns:
        compiled_model (RBLNCompiledModel): Compiled model that can be run on the RBLN NPU
    """
    mod = torchscript_to_ir(mod, input_names=input_names, input_info=input_info)
    compiled_model = compile(mod, **kwargs)
    return compiled_model
