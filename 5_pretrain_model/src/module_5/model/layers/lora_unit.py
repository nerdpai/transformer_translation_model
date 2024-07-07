import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass

from module_5.model.layers.layer_specs import LayerSpecs


@dataclass(frozen=True)
class LoraUnitSpecs:
    rank: int
    alpha: int


class LoraUnit(keras.layers.Layer):
    def __init__(
        self,
        shape: tf.TensorShape,
        specs: LoraUnitSpecs,
        layer_specs: LayerSpecs,
    ):
        super(LoraUnit, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )
        self.rank = specs.rank
        self.alpha = specs.alpha
        self.scale = specs.alpha / specs.rank
        self.shape = shape

        self.A: tf.Variable
        self.B: tf.Variable

    def build(self, input_shape) -> None:
        super().build(input_shape)

        input_d = self.shape[0]
        output_d = self.shape[1]

        initializer_A = keras.initializers.RandomNormal()
        initializer_B = keras.initializers.Zeros()
        initialized_A = initializer_A(shape=(input_d, self.rank), dtype=self.dtype)
        initialized_B = initializer_B(shape=(self.rank, output_d), dtype=self.dtype)
        self.A = tf.Variable(initialized_A, trainable=self.trainable)
        self.B = tf.Variable(initialized_B, trainable=self.trainable)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        output = tf.matmul(self.A, self.B)
        return output
