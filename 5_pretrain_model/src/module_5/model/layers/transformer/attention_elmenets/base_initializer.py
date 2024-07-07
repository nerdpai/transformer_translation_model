import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass, replace

from module_5.model.layers.lora import Lora, LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs

# pylint: disable=e0401
from module_5.model.constants import WEIGHTS_INITIALIZER, BIAS_INITIALIZER


@dataclass(frozen=True)
class BaseInitSpecs:
    emb_dim: int
    heads_num: int


class BaseInitializer(keras.layers.Layer):
    def __init__(
        self,
        specs: BaseInitSpecs,
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        self.layer_specs = layer_specs
        self.lora_specs = lora_specs
        super(BaseInitializer, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        if specs.emb_dim % specs.heads_num != 0:  # type: ignore
            raise ValueError("The number of heads must divisible by emb_dim")

        self.specs = specs

        self.parameters: tf.Variable
        self.biases: tf.Variable
        self.lora: Lora

    def build(self, input_shape) -> None:
        super().build(input_shape)

        weights_initializer = WEIGHTS_INITIALIZER
        bias_initializer = BIAS_INITIALIZER
        shape = tf.TensorShape((self.specs.emb_dim, self.specs.emb_dim))
        bias_shape = (self.specs.emb_dim,)

        weights = weights_initializer(shape=shape, dtype=self.dtype)
        self.parameters = tf.Variable(
            weights, trainable=self.trainable, dtype=self.dtype
        )
        biases = bias_initializer(shape=bias_shape, dtype=self.dtype)
        self.biases = tf.Variable(biases, trainable=self.trainable, dtype=self.dtype)
        self.lora = Lora(
            self.lora_specs,
            shape,
            replace(self.layer_specs, name=f"{self.layer_specs.name}_lora"),
        )
        self.lora.build(input_shape)

    def _prepare_call(self, condition: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        q_dim_for_head: int = self.specs.emb_dim // self.specs.heads_num

        transformer_weights = tf.add(self.parameters, self.lora(condition))
        transformer_weights = tf.reshape(
            transformer_weights,
            [self.specs.heads_num, q_dim_for_head, self.specs.emb_dim],
        )  # h*q*e

        transformer_biases = tf.reshape(
            self.biases, [self.specs.heads_num, q_dim_for_head]
        )
        transformer_biases = tf.expand_dims(transformer_biases, axis=0)  # 1*h*q

        return transformer_weights, transformer_biases
