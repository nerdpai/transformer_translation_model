import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import replace
from typing import Union


from module_5.model.layers.lora import Lora, LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs


class FullyConnected(keras.layers.Layer):
    def __init__(
        self,
        shape: tf.TensorShape,
        kernel_initializer: keras.initializers.Initializer,
        bias_initializer: Union[keras.initializers.Initializer, None],
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        self.lora_specs = lora_specs
        self.layer_specs = layer_specs
        super(FullyConnected, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        self.shape = shape
        self.input_dim = shape[0]
        self.output_dim = shape[1]
        self.kernel_init = kernel_initializer
        self.bias_init = bias_initializer
        self.use_bias = bias_initializer is not None

        if not self.use_bias:
            self.bias_init = keras.initializers.Zeros()

        self.dense: keras.layers.Dense
        self.lora: Lora

    def build(self, input_shape):
        super().build(input_shape)

        self.dense = keras.layers.Dense(  # this little gangsta is the one that uses 'cast' if you provide dtype (so no tflite), i don't really want to change it to the variables rn, so... good luck i guess? (now you can use only float32)
            units=self.output_dim,
            input_dim=self.input_dim,  # type: ignore
            bias_initializer=self.bias_init,  # type: ignore
            kernel_initializer=self.kernel_init,  # type: ignore
        )
        self.dense.build(tf.TensorShape((None, self.input_dim)))

        self.lora = Lora(
            self.lora_specs,
            self.shape,
            replace(self.layer_specs, name=f"{self.layer_specs.name}_lora"),
        )
        self.lora.build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        x, cond = inputs  # bs*??*input_dim, bs*1
        lora = self.lora(cond)
        x = tf.add(self.dense(x), tf.matmul(x, lora))
        return x
