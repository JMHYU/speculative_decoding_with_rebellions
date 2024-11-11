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

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rebel
import torch

from .__version__ import __version__
from .utils.runtime_utils import ContextRblnConfig


DEFAULT_COMPILED_MODEL_NAME = "compiled_model"
DEFAULT_MOD_NAME = "default"


@dataclass
class RBLNCompileConfig:
    """
    Configuration for RBLN compilation.

    Attributes:
        compiled_model_name (str): Name of the compiled model.
        mod_name (str): Name of the RBLN module.
        input_info (List[Tuple[str, Tuple[int], Optional[str]]]): Information about input tensors.
        fusion (Optional[bool]): Whether to use fusion optimization.
        npu (Optional[str]): NPU configuration.
        tensor_parallel_size (Optional[int]): Size for tensor parallelism.
    """

    compiled_model_name: str = DEFAULT_COMPILED_MODEL_NAME
    mod_name: str = DEFAULT_MOD_NAME
    input_info: List[Tuple[str, Tuple[int], Optional[str]]] = None
    fusion: Optional[bool] = None
    npu: Optional[str] = None
    tensor_parallel_size: Optional[int] = None

    @staticmethod
    def normalize_dtype(dtype):
        """
        Convert framework-specific dtype to string representation.
        i.e. torch.float32 -> "float32"

        Args:
            dtype: The input dtype (can be string, torch dtype, or numpy dtype).

        Returns:
            str: The normalized string representation of the dtype.
        """
        if isinstance(dtype, str):
            return dtype
        else:
            dtype: str = repr(dtype).split(".")[-1]
            if dtype.endswith("'>"):  # numpy
                dtype = dtype[:-2]
            return dtype

    def __post_init__(self):
        self.input_info = [(i[0], i[1], RBLNCompileConfig.normalize_dtype(i[2]) or "float32") for i in self.input_info]

    def update(self, kwargs: Dict[str, Any]):
        self.compiled_model_name = kwargs.get("compiled_model_name", self.compiled_model_name)
        self.mod_name = kwargs.get("mod_name", self.mod_name)
        self.input_info = kwargs.get("input_info", self.input_info)
        self.fusion = kwargs.get("fusion", self.fusion)
        self.npu = kwargs.get("npu", self.npu)
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", self.tensor_parallel_size)
        return self

    def get_dummy_inputs(self, fill=0):
        dummy = []
        for name, shape, dtype in self.input_info:
            dummy.append(
                torch.fill(torch.zeros(*shape, dtype=getattr(torch, dtype)), fill)
                if len(shape) > 0
                else torch.tensor(fill, dtype=getattr(torch, dtype))
            )
        return tuple(dummy)

    def asdict(self):
        return asdict(self)


RUNTIME_KEYWORDS = ["create_runtimes", "optimize_host_memory", "device", "device_map"]
COMPILE_KEYWORDS = ["compiled_model_name", "mod_name", "input_info", "fusion", "npu", "tensor_parallel_size"]


class RBLNConfig:
    """
    Configuration for single RBLN OptimizedModel, representing multiple compiled models.

    Attributes:
        compile_cfgs (List[RBLNCompileConfig]): Compilation configurations.
        meta (dict): Metadata including version and class information.
        runtime_cfg (dict): Runtime-specific configuration.
    """

    # It represents multiple compiled model, one of each can have multiple runtimes.
    def __init__(
        self,
        rbln_cls,
        compile_cfgs: List[RBLNCompileConfig],
        rbln_kwargs=None,
        meta=None,
    ) -> None:
        if rbln_kwargs is None:
            rbln_kwargs = {}
        else:
            rbln_kwargs = copy.deepcopy(rbln_kwargs)

        # meta : class, version and other informations.
        if meta is None:
            self.meta = {"version": __version__, "cls": rbln_cls}
        else:
            self.meta = meta

        # compile_cfgs : compile args for each runtime
        self.compile_cfgs = compile_cfgs
        for compile_cfg in self.compile_cfgs:
            compile_cfg.update(rbln_kwargs)
        for K in COMPILE_KEYWORDS:
            rbln_kwargs.pop(K, None)

        # runtime_cfg : Values that don't be saved / loaded.
        self.runtime_cfg = {}
        for runtime_key in RUNTIME_KEYWORDS:
            if runtime_key in rbln_kwargs:
                self.runtime_cfg[runtime_key] = rbln_kwargs.pop(runtime_key)

        # model_cfg : All user-provided values such as "max_seq_len".
        self.model_cfg: Dict[str, Any] = rbln_kwargs

    def save(self, dir_path: str):
        dir_path = Path(dir_path)

        s_json = {}
        compile_cfgs = [asdict(cfg) for cfg in self.compile_cfgs]
        s_json["_compile_cfgs"] = compile_cfgs
        s_json["_meta"] = self.meta
        s_json.update(self.model_cfg)

        with open(dir_path / "rbln_config.json", "w") as jsonf:
            json.dump(s_json, jsonf, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "RBLNConfig":
        dir_path = Path(dir_path)
        with open(dir_path / "rbln_config.json", "r") as jsonf:
            config_file = json.load(jsonf)

        return cls.fromdict(config_file)

    @classmethod
    def fromdict(cls, dic: dict):
        compile_cfgs = dic.pop("_compile_cfgs")
        compile_cfgs = [RBLNCompileConfig(**cfg) for cfg in compile_cfgs]

        meta = dic.pop("_meta")
        rbln_cls = meta["cls"]

        rbln_kwargs = dic
        return cls(rbln_cls=rbln_cls, compile_cfgs=compile_cfgs, rbln_kwargs=rbln_kwargs, meta=meta)

    def update_runtime_cfg(self, rbln_kwargs: Dict[str, Any]):
        keys = list(rbln_kwargs.keys())
        for key in keys:
            if key in RUNTIME_KEYWORDS:
                self.runtime_cfg[key] = rbln_kwargs[key]

    def __repr__(self):
        compile_cfgs_repr = [f"\n    {cfg!r}" for cfg in self.compile_cfgs]
        return (
            f"RBLNConfig(\n"
            f"  rbln_cls={self.meta['cls']},\n"
            f"  version='{self.meta['version']}',\n"
            f"  compile_cfgs=[{''.join(compile_cfgs_repr)}\n  ],\n"
            f"  model_cfg={self.model_cfg},\n"
            f"  runtime_cfg={self.runtime_cfg}\n"
            f")"
        )

    @property
    def create_runtimes(self):
        context = ContextRblnConfig.get_current_context()["create_runtimes"]
        if context is not None:
            return context
        elif self.runtime_cfg.get("create_runtimes", None) is None:
            return rebel.npu_is_available()
        return self.runtime_cfg["create_runtimes"]

    @property
    def optimize_host_memory(self):
        context = ContextRblnConfig.get_current_context()["optimize_host_memory"]
        if context is not None:
            return context
        elif self.runtime_cfg.get("optimize_host_memory", None) is None:
            return True
        return self.runtime_cfg["optimize_host_memory"]

    @property
    def device(self):
        context = ContextRblnConfig.get_current_context()["device"]
        if context:
            return context
        elif self.runtime_cfg.get("device", None) is None:
            return 0
        return self.runtime_cfg["device"]

    @property
    def device_map(self):
        context = ContextRblnConfig.get_current_context()["device_map"]
        if context:
            return context
        elif self.runtime_cfg.get("device_map", None) is None:
            rbln_device_map = {}
            device_val = self.device
            for cfg in self.compile_cfgs:
                rbln_device_map[cfg.compiled_model_name] = device_val
            return rbln_device_map
        return self.runtime_cfg["device_map"]
