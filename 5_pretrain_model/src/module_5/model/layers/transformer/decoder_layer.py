# only left to right
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras

from module_5.model.layers.transformer.encoder_layer import EncoderLayer
from module_5.model.layers.transformer.cross_attention import (
    CrossAttention,
    AttentionSpecs,
)
from module_5.model.layers.lora import LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs

# pylint: disable=e0401
from module_5.model.constants import NORM_EPSILON


class DecoderLayer(EncoderLayer):
    def __init__(
        self,
        specs: AttentionSpecs,
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        super(DecoderLayer, self).__init__(specs, lora_specs, layer_specs)

        self.cross_attenion: CrossAttention
        self.encoder_atten_norm: keras.layers.LayerNormalization

    def build(self, input_shape):
        super().build(input_shape)

        self.cross_attenion = CrossAttention(
            self.atten_specs,
            self.lora_specs,
            self._add_sufix(self.layer_specs, "cross_attention"),
        )
        self.cross_attenion.build(input_shape)

        self.encoder_atten_norm = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        *args,
        **kwargs,
    ) -> tf.Tensor:
        encoded, x, mask, condition = inputs  # bs*s*e, bs*s*e, bs*s, bs*1
        x = self.attention_part(x, mask, condition)  # bs*s*e

        hidden = self.encoder_atten_norm(x)  # bs*s*e
        hidden = self.cross_attenion([encoded, hidden, mask, condition])  # bs*s*e
        x = x + hidden  # bs*s*e

        x = self.fully_connected_part(x, condition)  # bs*s*e
        return x  # bs*s*e
