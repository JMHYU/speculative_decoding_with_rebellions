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


from .auto import (
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
)
from .bart import RBLNBartModel
from .bert import RBLNBertModel
from .clip import RBLNCLIPTextModel, RBLNCLIPTextModelWithProjection, RBLNCLIPVisionModel
from .dpt import RBLNDPTForDepthEstimation
from .gemma import RBLNGemmaForCausalLM
from .gpt2 import RBLNGPT2LMHeadModel
from .llama import RBLNLlamaForCausalLM
from .llava_next import RBLNLlavaNextForConditionalGeneration
from .midm import RBLNMidmLMHeadModel
from .mistral import RBLNMistralForCausalLM
from .phi import RBLNPhiForCausalLM
from .wav2vec2 import RBLNWav2Vec2ForCTC
from .whisper import RBLNWhisperForConditionalGeneration
from .xlm_roberta import RBLNXLMRobertaModel
