# only left to right
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import replace

from module_5.model.layers.transformer.encoder import CoderSpecs, get_builded_coder
from module_5.model.layers.transformer.decoder_layer import (
    DecoderLayer,
    AttentionSpecs,
)
from module_5.model.layers.transformer.positional_emb import PositionalEmbedding
from module_5.model.layers.lora import LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs

# pylint: disable=e0401
from module_5.model.constants import NORM_EPSILON


class Decoder(keras.layers.Layer):
    def __init__(
        self,
        specs: CoderSpecs,
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        self.atten_specs: AttentionSpecs = specs
        self.lora_specs = lora_specs
        self.layer_specs = layer_specs
        super(Decoder, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        self.specs = specs

        self.pos_emb: PositionalEmbedding
        self.encoder_layers: list[DecoderLayer]
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
                DecoderLayer,
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
        self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], *args, **kwargs
    ) -> tf.Tensor:
        encoded, x, mask, condition = inputs
        x = self.pos_emb([x, condition])
        x = self.positional_norm(x)
        for i in range(self.specs.coder_layers_num):
            x = self.encoder_layers[i]([encoded, x, mask, condition])

        return self.final_norm(x)
