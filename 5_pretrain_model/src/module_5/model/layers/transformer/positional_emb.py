import dataclasses
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras


from module_5.model.layers.lora import Lora, LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs

# pylint: disable=e0401
from module_5.model.constants import EMBEDDING_INITIALIZER


class PositionalEmbedding(keras.layers.Layer):
    def __init__(
        self,
        shape: tf.TensorShape,
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        self.layer_specs = layer_specs
        super(PositionalEmbedding, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        self.shape = shape
        self.lora_specs = lora_specs

        self.pos_embedding: tf.Variable
        self.lora: Lora

    def build(self, input_shape):
        super().build(input_shape)
        emb = EMBEDDING_INITIALIZER(shape=self.shape, dtype=self.dtype)
        self.pos_embedding = tf.Variable(
            emb, trainable=self.trainable, dtype=self.dtype
        )

        self.lora = Lora(
            self.lora_specs,
            self.shape,
            dataclasses.replace(self.layer_specs, name=f"{self.layer_specs.name}_lora"),
        )
        self.lora.build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        x, condition = inputs  # bs*s*e, bs*1
        lora = self.lora(condition)
        lora_emb = tf.add(self.pos_embedding, lora)
        return tf.add(x, lora_emb)  # bs*s*e
