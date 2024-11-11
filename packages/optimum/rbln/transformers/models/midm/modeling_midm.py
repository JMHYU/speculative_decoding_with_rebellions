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

from ....modeling_config import RBLNConfig
from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM
from .hf_hub_cached.modeling_midm import MidmLMHeadModel
from .midm_architecture import (
    MidmLMHeadModelWrapper,
)


logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from transformers import (
        PreTrainedModel,
    )


class RBLNMidmLMHeadModel(RBLNDecoderOnlyModelForCausalLM):
    """
    The Midm Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the
    library implements for all its model.

    It implements the methods to convert a pre-trained transformers Midm model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    """

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        rbln_max_seq_len = rbln_config.model_cfg["max_seq_len"]
        return MidmLMHeadModelWrapper(model, rbln_max_seq_len).eval()

    def __getattr__(self, __name: str) -> Any:
        """This is the key method to implement RBLN-Midm.

        Returns:
            Any: Midm's corresponding method
        """

        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(MidmLMHeadModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val
