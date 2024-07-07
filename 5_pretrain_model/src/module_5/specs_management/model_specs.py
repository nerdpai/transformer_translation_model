import tensorflow._api.v2.v2 as tf
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpecs:
    name: str
    transformer_layers_num: int
    heads_num: int
    seq_len: int
    lora_num: int
    lora_rank: int
    lora_alpha: int
    causal: bool
    dtype: tf.DType
