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

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

from .__version__ import __version__
from .utils import check_version_compats


_import_structure = {
    "modeling_alias": [
        "RBLNASTForAudioClassification",
        "RBLNBertForQuestionAnswering",
        "RBLNDistilBertForQuestionAnswering",
        "RBLNResNetForImageClassification",
        "RBLNT5ForConditionalGeneration",
        "RBLNBartForConditionalGeneration",
        "RBLNXLMRobertaForSequenceClassification",
        "RBLNRobertaForSequenceClassification",
        "RBLNRobertaForMaskedLM",
    ],
    "modeling_base": [
        "RBLNBaseModel",
        "RBLNModel",
        "RBLNModelForQuestionAnswering",
        "RBLNModelForAudioClassification",
        "RBLNModelForImageClassification",
        "RBLNModelForSequenceClassification",
        "RBLNModelForMaskedLM",
    ],
    "modeling_seq2seq": [
        "RBLNModelForSeq2SeqLM",
    ],
    "transformers": [
        "BatchTextIteratorStreamer",
        "RBLNAutoModel",
        "RBLNAutoModelForAudioClassification",
        "RBLNAutoModelForCausalLM",
        "RBLNAutoModelForCTC",
        "RBLNAutoModelForDepthEstimation",
        "RBLNAutoModelForImageClassification",
        "RBLNAutoModelForMaskedLM",
        "RBLNAutoModelForQuestionAnswering",
        "RBLNAutoModelForSeq2SeqLM",
        "RBLNAutoModelForSequenceClassification",
        "RBLNAutoModelForSpeechSeq2Seq",
        "RBLNAutoModelForVision2Seq",
        "RBLNBartModel",
        "RBLNBertModel",
        "RBLNCLIPTextModel",
        "RBLNCLIPTextModelWithProjection",
        "RBLNCLIPVisionModel",
        "RBLNDPTForDepthEstimation",
        "RBLNGemmaForCausalLM",
        "RBLNGPT2LMHeadModel",
        "RBLNWav2Vec2ForCTC",
        "RBLNLlamaForCausalLM",
        "RBLNPhiForCausalLM",
        "RBLNLlavaNextForConditionalGeneration",
        "RBLNMidmLMHeadModel",
        "RBLNMistralForCausalLM",
        "RBLNWhisperForConditionalGeneration",
        "RBLNXLMRobertaModel",
    ],
    "diffusers": [
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNAutoencoderKL",
        "RBLNUNet2DConditionModel",
        "RBLNControlNetModel",
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNMultiControlNetModel",
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
    ],
    "modeling_config": ["RBLNCompileConfig", "RBLNConfig"],
}

if TYPE_CHECKING:
    from .diffusers import (
        RBLNAutoencoderKL,
        RBLNControlNetModel,
        RBLNMultiControlNetModel,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionPipeline,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
        RBLNStableDiffusionXLImg2ImgPipeline,
        RBLNStableDiffusionXLPipeline,
        RBLNUNet2DConditionModel,
    )
    from .modeling_alias import (
        RBLNASTForAudioClassification,
        RBLNBartForConditionalGeneration,
        RBLNBertForQuestionAnswering,
        RBLNResNetForImageClassification,
        RBLNRobertaForMaskedLM,
        RBLNRobertaForSequenceClassification,
        RBLNT5ForConditionalGeneration,
        RBLNXLMRobertaForSequenceClassification,
    )
    from .modeling_base import (
        RBLNBaseModel,
        RBLNModel,
        RBLNModelForAudioClassification,
        RBLNModelForImageClassification,
        RBLNModelForMaskedLM,
        RBLNModelForQuestionAnswering,
        RBLNModelForSequenceClassification,
    )
    from .modeling_config import RBLNCompileConfig, RBLNConfig
    from .modeling_seq2seq import RBLNModelForSeq2SeqLM
    from .transformers import (
        BatchTextIteratorStreamer,
        RBLNAutoModel,
        RBLNAutoModelForAudioClassification,
        RBLNAutoModelForCausalLM,
        RBLNAutoModelForCTC,
        RBLNAutoModelForDepthEstimation,
        RBLNAutoModelForImageClassification,
        RBLNAutoModelForMaskedLM,
        RBLNAutoModelForQuestionAnswering,
        RBLNAutoModelForSeq2SeqLM,
        RBLNAutoModelForSequenceClassification,
        RBLNAutoModelForSpeechSeq2Seq,
        RBLNAutoModelForVision2Seq,
        RBLNBartModel,
        RBLNBertModel,
        RBLNCLIPTextModel,
        RBLNCLIPTextModelWithProjection,
        RBLNCLIPVisionModel,
        RBLNDPTForDepthEstimation,
        RBLNGemmaForCausalLM,
        RBLNGPT2LMHeadModel,
        RBLNLlamaForCausalLM,
        RBLNLlavaNextForConditionalGeneration,
        RBLNMidmLMHeadModel,
        RBLNMistralForCausalLM,
        RBLNPhiForCausalLM,
        RBLNWav2Vec2ForCTC,
        RBLNWhisperForConditionalGeneration,
        RBLNXLMRobertaModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )


check_version_compats()
