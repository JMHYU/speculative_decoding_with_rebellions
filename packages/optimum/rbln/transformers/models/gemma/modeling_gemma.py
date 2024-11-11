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

import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable

from transformers import GemmaForCausalLM

from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM
from .gemma_architecture import GemmaWrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ....modeling_config import RBLNConfig

logger = logging.getLogger(__name__)


class RBLNGemmaForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Gemma Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNMultiModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based GemmaForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers GemmaForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        rbln_max_seq_len = rbln_config.model_cfg["max_seq_len"]
        return GemmaWrapper(model, rbln_max_seq_len).eval()

    # AttributeError가 발생한 후 fallback으로 처리하기 위한 용도
    # model.generate에서 generate를 찾을 수 없었던 이유도 이것 때문인 것으로 추정. generate를 호출하면 이 함수가 호출된다.
    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(GemmaForCausalLM, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val
