# bidirectional
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from typing import Callable
from dataclasses import replace


from module_5.model.layers.transformer.self_attention import (
    SelfAttention,
    AttentionSpecs,
)
from module_5.model.layers.lora import LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs
from module_5.model.layers.fully_connect import FullyConnected

# pylint: disable=e0401
from module_5.model.constants import (
    WEIGHTS_INITIALIZER,
    BIAS_INITIALIZER,
    DIM_BOOM_FACTOR,
    NORM_EPSILON,
)


class EncoderLayer(keras.layers.Layer):
    def __init__(
        self,
        specs: AttentionSpecs,
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        self.lora_specs = lora_specs
        self.layer_specs = layer_specs
        super(EncoderLayer, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        self.atten_specs = specs
        self.dim = specs.emb_dim

        self.self_attention: SelfAttention
        self.atten_norm: keras.layers.LayerNormalization
        self.final_norm: keras.layers.LayerNormalization
        self.activation: Callable
        self.raise_dim: FullyConnected
        self.fall_dim: FullyConnected

    def _add_sufix(self, specs: LayerSpecs, sufix: str) -> LayerSpecs:
        return replace(specs, name=f"{specs.name}_{sufix}")

    def build(self, input_shape):
        super().build(input_shape)
        dim, boom_dim = self.dim, self.dim * DIM_BOOM_FACTOR

        self.self_attention = SelfAttention(
            self.atten_specs,
            self.lora_specs,
            self._add_sufix(self.layer_specs, "self_attention"),
        )
        self.self_attention.build(input_shape)

        self.atten_norm = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)
        self.final_norm = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)

        self.activation = keras.activations.gelu

        self.raise_dim = FullyConnected(
            tf.TensorShape([dim, boom_dim]),
            WEIGHTS_INITIALIZER,
            BIAS_INITIALIZER,
            self.lora_specs,
            self._add_sufix(self.layer_specs, "raise_dim"),
        )
        self.raise_dim.build(input_shape)

        self.fall_dim = FullyConnected(
            tf.TensorShape([boom_dim, dim]),
            WEIGHTS_INITIALIZER,
            BIAS_INITIALIZER,
            self.lora_specs,
            self._add_sufix(self.layer_specs, "fall_dim"),
        )
        self.fall_dim.build(input_shape)

    def attention_part(self, x, mask, condition):
        hidden = self.atten_norm(x)  # bs*s*e
        hidden = self.self_attention([hidden, mask, condition])  # bs*s*e
        hidden = hidden + x  # bs*s*e
        x = hidden  # bs*s*e
        return x  # bs*s*e

    def fully_connected_part(self, x, condition):
        hidden = self.final_norm(x)  # bs*s*e
        hidden = self.raise_dim([hidden, condition])  # bs*s*boom_dim
        hidden = self.activation(hidden)  # bs*s*boom_dim
        hidden = self.fall_dim([hidden, condition])  # bs*s*e
        hidden = hidden + x  # bs*s*e
        x = hidden  # bs*s*e
        return x  # bs*s*e

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        *args,
        **kwargs,
    ) -> tf.Tensor:
        x, mask, condition = inputs  # bs*s*e, bs*s, bs*1
        x = self.attention_part(x, mask, condition)  # bs*s*e
        x = self.fully_connected_part(x, condition)  # bs*s*e
        return x  # bs*s*e
