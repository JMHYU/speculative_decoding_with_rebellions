from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class MidmBitextConfig(GPT2Config):
    model_type = "midm-bitext-S"

    def __init__(
        self,
        use_absolute_position_embedding: bool = True,
        use_rotary_position_embedding: bool = False,
        rotary_percentage: float = 1.0,
        normalization_type: str = "layernorm",
        scale_qk_by_inverse_layer_idx: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_absolute_position_embedding = use_absolute_position_embedding
        self.use_rotary_position_embedding = use_rotary_position_embedding
        self.rotary_percentage = rotary_percentage
        self.normalization_type = normalization_type
        self.scale_qk_by_inverse_layer_idx = scale_qk_by_inverse_layer_idx
