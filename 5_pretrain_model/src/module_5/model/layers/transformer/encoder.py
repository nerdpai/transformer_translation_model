# bidirectional
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass, replace


from module_5.model.layers.transformer.encoder_layer import (
    EncoderLayer,
    AttentionSpecs,
)
from module_5.model.layers.transformer.positional_emb import PositionalEmbedding
from module_5.model.layers.lora import LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs

# pylint: disable=e0401
from module_5.model.constants import NORM_EPSILON


def get_builded_coder(
    t: type,
    input_shape,
    atten_specs: AttentionSpecs,
    lora_specs: LoraSpecs,
    layer_specs: LayerSpecs,
    i: int,
) -> keras.layers.Layer:
    coder = t(
        atten_specs,
        lora_specs,
        replace(layer_specs, name=f"{layer_specs.name}_layer_{i}"),
    )
    coder.build(input_shape)
    return coder


@dataclass(frozen=True)
class CoderSpecs(AttentionSpecs):
    coder_layers_num: int


class Encoder(keras.layers.Layer):
    def __init__(
        self,
        specs: CoderSpecs,
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        self.atten_specs: AttentionSpecs = specs
        self.lora_specs = lora_specs
        self.layer_specs = layer_specs
        super(Encoder, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        self.specs = specs

        self.pos_emb: PositionalEmbedding
        self.encoder_layers: list[EncoderLayer]
        self.positional_norm: keras.layers.LayerNormalization
        self.final_norm: keras.layers.LayerNormalization

    def build(self, input_shape):
        super().build(input_shape)
        pos_shape = tf.TensorShape((self.specs.seq_len, self.specs.emb_dim))

        self.pos_emb = PositionalEmbedding(
            pos_shape,
            self.lora_specs,
            replace(self.layer_specs, name=f"{self.layer_specs.name}_pos_emb"),
        )
        self.pos_emb.build(input_shape)

        self.encoder_layers = [
            get_builded_coder(
                EncoderLayer,
                input_shape,
                self.atten_specs,
                self.lora_specs,
                self.layer_specs,
                i,
            )
            for i in range(self.specs.coder_layers_num)
        ]

        self.positional_norm = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)
        self.final_norm = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], *args, **kwargs
    ) -> tf.Tensor:
        x, mask, condition = inputs  # bs*s*e, bs*s, bs*1
        x = self.pos_emb([x, condition])  # bs*s*e
        x = self.positional_norm(x)  # bs*s*e
        for i in range(self.specs.coder_layers_num):
            x = self.encoder_layers[i]([x, mask, condition])  # bs*s*e

        return self.final_norm(x)
