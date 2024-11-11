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
import importlib
import inspect
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import rebel
import torch
import transformers
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    GenerationConfig,
    PretrainedConfig,
)

from .modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from .utils.runtime_utils import UnavailableRuntime
from .utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors


if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PreTrainedModel,
    )

logger = logging.getLogger(__name__)


class SubModulesMixin:
    """
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]
    """

    _rbln_submodules: List[Dict[str, Any]] = []

    def __init__(
        self,
        *,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ) -> None:
        for submodule_meta, submodule in zip(self._rbln_submodules, rbln_submodules):
            setattr(self, submodule_meta["name"], submodule)

    @classmethod
    def _from_model(
        cls,
        model: "PreTrainedModel",
        model_save_dir: str,
        rbln_sub_configs_dict: Dict[str, Any],
        rbln_kwargs: Dict[str, Any],
        subfolder=None,  # warning: will be ignored
        **kwargs,
    ) -> List["RBLNBaseModel"]:
        rbln_submodules = []
        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]
            torch_submodule: "PreTrainedModel" = getattr(model, submodule["name"])
            cls_name = torch_submodule.__class__.__name__
            submodule_cls: "RBLNBaseModel" = getattr(importlib.import_module("optimum.rbln"), f"RBLN{cls_name}")

            if submodule_name in rbln_sub_configs_dict:
                kwargs["rbln_config"] = rbln_sub_configs_dict[submodule_name]

            rbln_submodule = submodule_cls._export(
                model_id=None,
                config=torch_submodule.config,
                subfolder=submodule_name,
                model_save_dir=model_save_dir,
                model=torch_submodule,
                **rbln_kwargs,
                **kwargs,
            )

            rbln_submodules.append(rbln_submodule)

        return rbln_submodules

    @classmethod
    def _submodule_from_compiled_model(
        cls, model_save_dir: str, rbln_sub_configs_dict: Dict[str, Any], rbln_kwargs: Dict[str, Any], **kwargs
    ):
        rbln_submodules = []
        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]
            rbln_submodule_config_dict = rbln_sub_configs_dict.get(submodule_name, None)

            # Get cls name for call the constructor of the rbln class
            submodule_rbln_config = RBLNConfig.load(Path(model_save_dir) / submodule_name)
            submodule_cls_name = submodule_rbln_config.meta["cls"]
            submodule_cls: "RBLNBaseModel" = getattr(importlib.import_module("optimum.rbln"), submodule_cls_name)

            config = OptimizedModel._load_config(Path(model_save_dir) / submodule_name, **kwargs)
            rbln_submodule = submodule_cls._from_pretrained(
                model_id=model_save_dir,
                config=config,
                subfolder=submodule_name,
                rbln_config=rbln_submodule_config_dict,
                **rbln_kwargs,
                **kwargs,
            )
            rbln_submodules.append(rbln_submodule)
        return rbln_submodules

    @classmethod
    def _load_submodules(
        cls,
        model_save_dir,
        rbln_sub_configs_dict,
        rbln_kwargs,
        model=None,
        **kwargs,
    ):
        # Two way :
        # 1. Compile from pytorch object
        # 2. Load from compiled file
        if model is not None:
            return cls._from_model(
                model=model,
                model_save_dir=model_save_dir,
                rbln_sub_configs_dict=rbln_sub_configs_dict,
                rbln_kwargs=rbln_kwargs,
                **kwargs,
            )

        else:
            return cls._submodule_from_compiled_model(
                model_save_dir=model_save_dir,
                rbln_sub_configs_dict=rbln_sub_configs_dict,
                rbln_kwargs=rbln_kwargs,
                **kwargs,
            )


class RBLNBaseModel(OptimizedModel, ABC, SubModulesMixin):
    """
    An abstract base class for compiling, loading, and saving neural network models from the huggingface
    transformers and diffusers libraries to run on RBLN NPU devices.

    This class supports loading and saving models using the `from_pretrained` and `save_pretrained` methods,
    similar to the huggingface libraries.

    The `from_pretrained` method loads a model corresponding to the given `model_id` from a local repository
    or the huggingface hub onto the NPU. If the model is a PyTorch model and `export=True` is passed as a
    kwarg, it compiles the PyTorch model corresponding to the given `model_id` before loading. If `model_id`
    is an already rbln-compiled model, it can be directly loaded onto the NPU with `export=False`.

    `rbln_npu` is a kwarg required for compilation, specifying the name of the NPU to be used. If this
    keyword is not specified, the NPU installed on the host machine is used. If no NPU is installed on the
    host machine, an error occurs.

    `rbln_device` specifies the device to be used at runtime. If not specified, device 0 is used.

    `rbln_create_runtimes` indicates whether to create runtime objects. If False, the runtime does not load
    the model onto the NPU. This option is particularly useful when you want to perform compilation only on a
    host machine without an NPU.

    `RBLNModel`, `RBLNModelFor*`, etc. are all child classes of RBLNBaseModel.

    Models compiled in this way can be saved to a local repository using `save_pretrained` or uploaded to
    the huggingface hub.

    It also supports generation through `generate` (for transformers models that support generation).

    RBLNBaseModel is a class for models consisting of an arbitrary number of `torch.nn.Module`s, and
    therefore is an abstract class without explicit implementations of `forward` or `export` functions.
    To inherit from this class, `forward`, `export`, etc. must be implemented.
    """

    model_type = "rbln_model"
    auto_model_class = AutoModel  # feature extraction
    config_name = "config.json"

    def __init__(
        self,
        models: List[rebel.Runtime],
        config: "PretrainedConfig",
        rbln_config: RBLNConfig,
        preprocessors: Optional[List],
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        rbln_compiled_models: Optional[rebel.RBLNCompiledModel] = None,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ):
        super().__init__(models, config)
        if not isinstance(self.config, PretrainedConfig):  # if diffusers config
            self.config = PretrainedConfig(**self.config)

        self.rbln_config = rbln_config
        self.preprocessors = [] if preprocessors is None else preprocessors
        self.compiled_models = rbln_compiled_models

        # Registers the RBLNBaseModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/3d3204c025b6b5de013e07dd364208e28b4d9589/src/transformers/pipelines/base.py#L940
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

        # copied from tranformers PreTrainedModel __init__
        if self.can_generate():
            gen_config_dir = model_save_dir.name if isinstance(model_save_dir, TemporaryDirectory) else model_save_dir
            self.generation_config = GenerationConfig.from_pretrained(gen_config_dir, trust_remote_code=True)
        else:
            self.generation_config = None

        # self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        if self.generation_config is not None:
            self.generation_config.use_cache = True

        self.device = torch.device("cpu")
        self.training = False

        # FIXME :: model_save_dir is not used after initialized. (This can be used when save/load)
        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying RBLN model.
        self._model_save_dir_tempdirectory_instance = None
        if isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir
        self.subfolder = subfolder

        self.rbln_submodules = rbln_submodules
        self.__post_init__(**kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.rbln.modeling_base.RBLNBaseModel.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        real_save_dir = self.model_save_dir / self.subfolder
        save_directory_path = Path(save_directory)
        if os.path.exists(real_save_dir) and os.path.isdir(real_save_dir):
            if save_directory_path.absolute() == real_save_dir.absolute():
                raise FileExistsError(
                    f"Cannot save model to '{save_directory}'. "
                    f"This directory already exists and contains the model files."
                )
            shutil.copytree(real_save_dir, save_directory, dirs_exist_ok=True)
            self.config.save_pretrained(save_directory)
            if self.generation_config is not None:
                self.generation_config.save_pretrained(save_directory)
        else:
            raise FileNotFoundError(
                f"Unable to save the model. The model directory '{real_save_dir}' does not exist or is not accessible. "
                f"Cannot save to the specified destination '{save_directory}'. "
                f"Please ensure the model directory exists and you have the necessary permissions to access it."
            )

    @classmethod
    def _load_compiled_model_dir(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
    ):
        # Find compiled model
        # And prepare or download cache folder from HF Hub if needed.
        model_path = Path(model_id)
        if model_path.is_dir():
            model_path = model_path / subfolder
            rbln_files = list(model_path.glob("*.rbln"))
            rbln_config_filenames = list(model_path.glob("rbln_config.json"))
        else:
            if isinstance(use_auth_token, bool):
                token = HfFolder().get_token()
            else:
                token = use_auth_token
            repo_files = list(
                map(
                    Path,
                    HfApi().list_repo_files(model_id, revision=revision, token=token),
                )
            )

            pattern = "*.rbln" if subfolder == "" else f"{subfolder}/*.rbln"
            rbln_files = [p for p in repo_files if p.match(pattern)]

            pattern = "rbln_config.json" if subfolder == "" else f"{subfolder}/rbln_config.json"
            rbln_config_filenames = [p for p in repo_files if p.match(pattern)]

        if len(rbln_files) == 0:
            raise FileNotFoundError(f"Could not find any rbln model file in {model_path}")

        if len(rbln_config_filenames) == 0:
            raise FileNotFoundError(f"Could not find `rbln_config.json` file in {model_path}")

        if len(rbln_config_filenames) > 1:
            raise FileExistsError(
                f"Multiple rbln_config.json are not expected. but {len(rbln_config_filenames)} are found."
            )

        if model_path.is_dir():
            model_path = str(model_path)
        else:
            rbln_config_filename = rbln_config_filenames[0]
            rbln_config_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=str(rbln_config_filename),
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            model_path = Path(rbln_config_cache_path).parent

        return model_path

    @classmethod
    def _load_compiled_models(cls, model_path: str):
        compiled_models = Path(model_path).glob("*.rbln")
        rbln_compiled_models = {cm.stem: rebel.RBLNCompiledModel(cm) for cm in compiled_models}
        return rbln_compiled_models

    @classmethod
    def _split_submodule_config(cls, rbln_config_dict: Dict[str, Any] = {}) -> Dict[str, Any]:
        # {"language_model" : {"rbln_tensor_parallel_size":4}}
        rbln_sub_configs_dict: Dict[str, Dict[str, Any]] = {}

        # Remove submodule-configs from rbln_config
        if len(cls._rbln_submodules) > 0:
            keys = list(rbln_config_dict.keys())
            submodule_names = [m["name"] for m in cls._rbln_submodules]
            for key in keys:
                if key in submodule_names:
                    rbln_sub_configs_dict[key] = rbln_config_dict.pop(key)

        return rbln_sub_configs_dict

    @classmethod
    def resolve_rbln_config(cls, rbln_config: Union[RBLNConfig, Dict[str, Any]], kwargs):
        if isinstance(rbln_config, RBLNConfig):
            # Already resolved
            return rbln_config, None

        else:
            if rbln_config is None:
                rbln_config_dict = {}
            else:
                rbln_config_dict = rbln_config

            rbln_kwargs = cls.pop_rbln_kwargs_from_kwargs(kwargs)
            rbln_sub_configs_dict = cls._split_submodule_config(rbln_config_dict)

            for key in rbln_config_dict:
                if key in rbln_kwargs:
                    raise KeyError(f"duplicate key in both `rbln_config` and {key}")

            merged_rbln_kwargs = copy.deepcopy(rbln_kwargs)
            merged_rbln_kwargs.update(rbln_config_dict)

            return (merged_rbln_kwargs, rbln_sub_configs_dict)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        # passed from compile function
        rbln_config: Optional[RBLNConfig] = None,
        rbln_compiled_models: Optional[Dict[str, rebel.RBLNCompiledModel]] = None,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ) -> "RBLNBaseModel":
        from_export_method = isinstance(rbln_config, RBLNConfig) and rbln_compiled_models is not None

        if not from_export_method:
            # from compiled dir
            rbln_kwargs, rbln_sub_configs_dict = cls.resolve_rbln_config(rbln_config, kwargs)

            model_path_subfolder = cls._load_compiled_model_dir(
                model_id=model_id,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                subfolder=subfolder,
                local_files_only=local_files_only,
            )

            rbln_config = RBLNConfig.load(model_path_subfolder)
            rbln_config.update_runtime_cfg(rbln_kwargs)

            rbln_compiled_models = cls._load_compiled_models(model_path_subfolder)

            if len(cls._rbln_submodules) > 0:
                rbln_submodules = cls._load_submodules(
                    model_save_dir=model_id,
                    rbln_sub_configs_dict=rbln_sub_configs_dict,
                    rbln_kwargs=rbln_kwargs,
                    **kwargs,
                )
            else:
                rbln_submodules = []

            if subfolder != "":
                model_save_dir = Path(model_path_subfolder).absolute().parent
            else:
                model_save_dir = Path(model_path_subfolder).absolute()

        return cls._from_compiled_models(
            rbln_compiled_models=rbln_compiled_models,
            rbln_config=rbln_config,
            config=config,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    def _from_compiled_models(
        cls,
        rbln_compiled_models: Dict[str, rebel.RBLNCompiledModel],
        rbln_config: RBLNConfig,
        config,
        model_save_dir: str,
        subfolder: str,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ):
        if isinstance(model_save_dir, str):
            model_save_dir = Path(model_save_dir)
        preprocessors = maybe_load_preprocessors(model_save_dir.name, subfolder=subfolder)

        # FIXME:: Should we convert it?
        compiled_model_names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        rbln_compiled_models = [rbln_compiled_models[cm_name] for cm_name in compiled_model_names]

        # create runtimes only if `rbln_create_runtimes` is enabled
        models = (
            cls._create_runtimes(rbln_compiled_models, rbln_config.device_map)
            if rbln_config.create_runtimes
            else UnavailableRuntime()
        )

        return cls(
            models,
            config,
            rbln_config,
            preprocessors,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
            rbln_compiled_models=(None if rbln_config.optimize_host_memory else rbln_compiled_models),
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    def __repr__(self):
        return repr(self.model) + repr(self.rbln_submodules)

    @classmethod
    def compile(cls, model, rbln_compile_config: Optional[RBLNCompileConfig] = None):
        compiled_model = rebel.compile_from_torch(
            model,
            input_info=rbln_compile_config.input_info,
            fusion=rbln_compile_config.fusion,
            npu=rbln_compile_config.npu,
            tensor_parallel_size=rbln_compile_config.tensor_parallel_size,
        )
        return compiled_model

    @classmethod
    def get_rbln_config(
        cls,
        rbln_kwargs: Dict[str, Any],
        **others,
    ) -> RBLNConfig:
        """
        Make default rbln-config for the model.
        kwargs for overriding model's config can be accepted.
        Note that batch_size should be specified with proper input_info.
        """
        rbln_config = cls._get_rbln_config(**others, rbln_kwargs=rbln_kwargs)
        return rbln_config

    @staticmethod
    def pop_rbln_kwargs_from_kwargs(kwargs: Dict[str, Any], runtime_only=False):
        keys = list(kwargs.keys())
        rbln_kwargs = {key[5:]: kwargs.pop(key) for key in keys if key.startswith("rbln_")}

        if runtime_only:
            rbln_kwargs = {
                key: value
                for key, value in rbln_kwargs.items()
                if key in {"create_runtimes", "optimize_host_memory", "device", "device_map"}
            }

        return rbln_kwargs

    def can_generate(self):
        return False

    def to(self, *args, **kwargs):
        # Do nothing
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __post_init__(self, **kwargs):
        self.dtype = torch.float32

    @classmethod
    def _from_transformers(cls, *args, **kwargs) -> "RBLNBaseModel":
        """
        Exports a vanilla Transformers model into a rbln-compiled Module.
        This will be deprecated after optimum 2.0
        """
        return cls._export(*args, **kwargs)

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        # Wrap the model if needed.
        return model

    @classmethod
    @abstractmethod
    def _get_rbln_config(cls, **rbln_config_kwargs) -> RBLNConfig:
        pass

    @abstractmethod
    def forward(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        pass

    @classmethod
    @abstractmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
    ) -> List[rebel.Runtime]:
        # compiled_models -> runtimes
        pass

    @classmethod
    @abstractmethod
    def _export(cls, *args, **kwargs):
        """
        Exports a vanilla Transformers model into a rbln-compiled Module.
        """
        pass


class RBLNModel(RBLNBaseModel):
    """
    A class that inherits from RBLNBaseModel for models consisting of a single `torch.nn.Module`.

    This class supports all the functionality of RBLNBaseModel, including loading and saving models using
    the `from_pretrained` and `save_pretrained` methods, compiling PyTorch models for execution on RBLN NPU
    devices.

    Example:
        ```python
        model = RBLNModel.from_pretrained("model_id", export=True, rbln_npu="npu_name")
        outputs = model(**inputs)
        ```
    """

    @classmethod
    def update_kwargs(cls, kwargs):
        """
        Update user-given kwargs to get proper pytorch model.

        For example, `torchscript`=True should be set because torch.jit
        does not support `transformers` output instances as module output;
        """
        kwargs.update(
            {
                "torchscript": True,
                "return_dict": False,
            }
        )
        return kwargs

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        # Some rbln-kwargs should be applied before loading torch module (i.e. quantized llm)
        rbln_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        task = kwargs.pop("task", None)
        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        kwargs = cls.update_kwargs(kwargs)

        model = TasksManager.get_model_from_task(
            task=task,
            model_name_or_path=model_id,
            subfolder=subfolder,
            revision=revision,
            framework="pt",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return model

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """

    @classmethod
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        model = cls.wrap_model_if_needed(model, rbln_config)
        rbln_compile_config = rbln_config.compile_cfgs[0]
        compiled_model = cls.compile(model, rbln_compile_config=rbln_compile_config)
        return compiled_model

    @classmethod
    @torch.no_grad()
    def _export(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        model: "PreTrainedModel" = None,
        rbln_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "RBLNModel":
        rbln_kwargs, rbln_sub_configs_dict = cls.resolve_rbln_config(rbln_config, kwargs)

        if model_save_dir is None:
            save_dir = TemporaryDirectory()
            save_dir_path = Path(save_dir.name)
        else:
            save_dir = model_save_dir
            if isinstance(save_dir, TemporaryDirectory):
                save_dir_path = Path(model_save_dir.name)
            else:
                save_dir_path = Path(model_save_dir)
                save_dir_path.mkdir(exist_ok=True)

        # Load pytorch model if needed.
        if model is None:
            model: "PreTrainedModel" = cls.get_pytorch_model(
                model_id=model_id,
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
                force_download=force_download,
                trust_remote_code=trust_remote_code,
                rbln_kwargs=rbln_kwargs,
                **kwargs,
            )
            preprocessors = maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)
        else:
            preprocessors = []

        # FIXME :: optimum passes AutoConfig.
        config = model.config
        if hasattr(model, "can_generate") and model.can_generate():
            generation_config = model.generation_config
            generation_config.save_pretrained(save_dir_path / subfolder)

        if not isinstance(config, PretrainedConfig):  # diffusers config
            config = PretrainedConfig(**config)
        config.save_pretrained(save_dir_path / subfolder)

        # Get compilation arguments
        rbln_config: RBLNConfig = cls.get_rbln_config(
            preprocessors=preprocessors, model_config=config, rbln_kwargs=rbln_kwargs
        )
        compiled_model: Union[rebel.RBLNCompiledModel, Dict[str, rebel.RBLNCompiledModel]] = cls.get_compiled_model(
            model, rbln_config=rbln_config
        )

        # Save compiled models
        (save_dir_path / subfolder).mkdir(exist_ok=True)
        if not isinstance(compiled_model, dict):
            compiled_models = {DEFAULT_COMPILED_MODEL_NAME: compiled_model}
        else:
            compiled_models = compiled_model
        for compiled_model_name, cm in compiled_models.items():
            cm.save(save_dir_path / subfolder / f"{compiled_model_name}.rbln")
        rbln_config.save(save_dir_path / subfolder)

        cls.save_torch_artifacts(model, save_dir_path=save_dir_path, subfolder=subfolder, rbln_config=rbln_config)

        # Load submodules
        if len(cls._rbln_submodules) > 0:
            rbln_submodules = cls._load_submodules(
                model=model,
                model_save_dir=save_dir,
                rbln_sub_configs_dict=rbln_sub_configs_dict,
                rbln_kwargs=rbln_kwargs,
                **kwargs,
            )
        else:
            rbln_submodules = []

        # Instantiate
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            model_save_dir=save_dir,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            subfolder=subfolder,
            local_files_only=local_files_only,
            rbln_config=rbln_config,
            rbln_compiled_models=compiled_models,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
    ) -> List[rebel.Runtime]:
        device = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
        return [compiled_model.create_runtime(tensor_type="pt", device=device) for compiled_model in compiled_models]

    def forward(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        output = self.model[0](*args, **kwargs)
        return output


class RBLNModelForQuestionAnswering(RBLNModel):
    model_type = "rbln_model"
    auto_model_class = AutoModelForQuestionAnswering
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)

        if rbln_max_seq_len is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_max_length"):
                    rbln_max_seq_len = tokenizer.model_max_length
                    break
            if rbln_max_seq_len is None:
                raise ValueError("`rbln_max_seq_len` should be specified!")

        if rbln_batch_size is None:
            rbln_batch_size = 1

        if rbln_model_input_names is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_input_names"):
                    rbln_model_input_names = tokenizer.model_input_names
                    break
            if rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names"):
                rbln_model_input_names = cls.rbln_model_input_names
            elif rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names") is False:
                original_model_class = getattr(transformers, model_config.architectures[0])
                input_names_order = inspect.signature(original_model_class.forward).parameters.keys()
                raise ValueError(
                    "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`, "
                    f"and be sure to make the order of the inputs same as QuestionAnswering forward() arguments like ({list(input_names_order)})"
                )

        input_info = [
            (model_input_name, [rbln_batch_size, rbln_max_seq_len], "int64")
            for model_input_name in rbln_model_input_names
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        rbln_config.model_cfg.update({"max_seq_len": rbln_max_seq_len})
        return rbln_config


class RBLNModelForImageClassification(RBLNModel):
    """
    This is a generic model class that will be instantiated as one of the model classes of the library (with a image classification head) when created with the from_pretrained() class method
    """

    model_type = "rbln_model"
    auto_model_class = AutoModelForImageClassification

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_image_size = rbln_kwargs.get("image_size", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        if rbln_image_size is None:
            for processor in preprocessors:
                if hasattr(processor, "size"):
                    rbln_image_size = processor.size["shortest_edge"]
                    break
            if rbln_image_size is None:
                raise ValueError("`rbln_image_size` should be specified!")

        if rbln_batch_size is None:
            rbln_batch_size = 1

        if isinstance(rbln_image_size, int):
            rbln_image_size = rbln_image_size, rbln_image_size

        input_info = [
            (
                "pixel_values",
                [rbln_batch_size, 3, rbln_image_size[0], rbln_image_size[1]],
                "float32",
            )
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        return RBLNConfig(rbln_cls=cls.__name__, compile_cfgs=[rbln_compile_config], rbln_kwargs=rbln_kwargs)


class RBLNModelForAudioClassification(RBLNModel):
    """
    This is a generic model class that will be instantiated as one of the model classes of the library (with a audio classification head) when created with the from_pretrained() class method
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based AudioClassification models on RBLN devices.
    It implements the methods to convert a pre-trained transformers AudioClassification model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Currently, this model class only supports the 'AST' model from the transformers library. Future updates may include support for additional model types.
    """

    model_type = "rbln_model"
    auto_model_class = AutoModelForAudioClassification

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: "AutoFeatureExtractor",
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_max_length = rbln_kwargs.get("max_length", None)
        rbln_num_mel_bins = rbln_kwargs.get("num_mel_bins", None)

        if rbln_batch_size is None:
            rbln_batch_size = 1

        if rbln_num_mel_bins is None:
            rbln_num_mel_bins = getattr(model_config, "num_mel_bins", None)
            if rbln_num_mel_bins is None:
                for feature_extractor in preprocessors:
                    if hasattr(feature_extractor, "num_mel_bins"):
                        rbln_num_mel_bins = feature_extractor.num_mel_bins
                        break

        if rbln_num_mel_bins is None:
            raise ValueError("`rbln_num_mel_bins` should be specified!")

        if rbln_max_length is None:
            rbln_max_length = getattr(model_config, "max_length", None)
            for feature_extractor in preprocessors:
                if hasattr(feature_extractor, "max_length"):
                    rbln_max_length = feature_extractor.max_length
                    break

        if rbln_max_length is None:
            raise ValueError("`rbln_max_length` should be specified!")

        input_info = [
            (
                "input_values",
                [rbln_batch_size, rbln_max_length, rbln_num_mel_bins],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        rbln_config.model_cfg.update(
            {
                "batch_size": rbln_batch_size,
                "max_length": rbln_max_length,
                "num_mel_bins": rbln_num_mel_bins,
            }
        )
        return rbln_config


class RBLNModelForSequenceClassification(RBLNModel):
    """
    This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence classification head) when created with the from_pretrained() class method
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based SequenceClassification models on RBLN devices.
    It implements the methods to convert a pre-trained transformers SequenceClassification model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Currently, this model class supports the 'XLMRoberta' and 'Roberta' model from the transformers library. Future updates may include support for additional model types.
    """

    model_type = "rbln_model"
    auto_model_class = AutoModelForSequenceClassification

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        max_position_embeddings = getattr(model_config, "n_positions", None) or getattr(
            model_config, "max_position_embeddings", None
        )

        if rbln_max_seq_len is None:
            rbln_max_seq_len = max_position_embeddings
            if rbln_max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_max_seq_len = tokenizer.model_max_length
                        break
                if rbln_max_seq_len is None:
                    raise ValueError("`rbln_max_seq_len` should be specified!")

        if max_position_embeddings is not None and rbln_max_seq_len > max_position_embeddings:
            raise ValueError("`rbln_enc_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_model_input_names is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_input_names"):
                    rbln_model_input_names = tokenizer.model_input_names
                    break
            if rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names"):
                rbln_model_input_names = cls.rbln_model_input_names
            elif rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names") is False:
                original_model_class = getattr(transformers, model_config.architectures[0])
                input_names_order = inspect.signature(original_model_class.forward).parameters.keys()
                raise ValueError(
                    "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`, "
                    f"and be sure to make the order of the inputs same as SequenceClassification forward() arguments like ({list(input_names_order)})"
                )

        if rbln_batch_size is None:
            rbln_batch_size = 1

        input_info = [
            (model_input_name, [rbln_batch_size, rbln_max_seq_len], "int64")
            for model_input_name in rbln_model_input_names
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        rbln_config.model_cfg.update({"max_seq_len": rbln_max_seq_len})
        return rbln_config


class RBLNModelForMaskedLM(RBLNModel):
    model_type = "rbln_model"
    auto_model_class = AutoModelForMaskedLM

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        max_position_embeddings = getattr(model_config, "n_positions", None) or getattr(
            model_config, "max_position_embeddings", None
        )

        if rbln_max_seq_len is None:
            rbln_max_seq_len = max_position_embeddings
            if rbln_max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_max_seq_len = tokenizer.model_max_length
                        break
                if rbln_max_seq_len is None:
                    raise ValueError("`rbln_max_seq_len` should be specified!")

        if max_position_embeddings is not None and rbln_max_seq_len > max_position_embeddings:
            raise ValueError("`rbln_enc_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_model_input_names is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_input_names"):
                    rbln_model_input_names = tokenizer.model_input_names
                    break
            if rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names"):
                rbln_model_input_names = cls.rbln_model_input_names
            elif rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names") is False:
                original_model_class = getattr(transformers, model_config.architectures[0])
                input_names_order = inspect.signature(original_model_class.forward).parameters.keys()
                raise ValueError(
                    "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`, "
                    f"and be sure to make the order of the inputs same as MaskedLM forward() arguments like ({list(input_names_order)})"
                )

        if rbln_batch_size is None:
            rbln_batch_size = 1

        input_info = [
            (model_input_name, [rbln_batch_size, rbln_max_seq_len], "int64")
            for model_input_name in rbln_model_input_names
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        rbln_config.model_cfg.update({"max_seq_len": rbln_max_seq_len})
        return rbln_config
