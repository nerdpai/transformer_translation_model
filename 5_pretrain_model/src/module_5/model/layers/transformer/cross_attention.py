import numpy as np
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass, replace
from typing import Callable


from module_5.model.layers.lora import LoraSpecs
from module_5.model.layers.transformer.attention_elmenets.extractor import Extractor
from module_5.model.layers.transformer.attention_elmenets.out_proj import OutProj
from module_5.model.layers.transformer.attention_elmenets.base_initializer import (
    BaseInitSpecs,
    BaseInitializer,
)
from module_5.model.layers.layer_specs import LayerSpecs


# def causal(logits: tf.Tensor) -> tf.Tensor:
#     # mask = tf.linalg.band_part(tf.ones_like(logits, tf.uint8), -1, 0)
#     last_dim = tf.shape(logits)[-1]
#     mask: tf.Tensor = tf.ones((last_dim,), dtype=tf.int32)
#     mask = tf.cumsum(mask)
#     mask = tf.expand_dims(mask, 0)
#     mask = tf.repeat(mask, last_dim, axis=0)
#     triangle = tf.range(last_dim, dtype=tf.int32)
#     triangle = tf.add(triangle, 1)
#     triangle = tf.expand_dims(triangle, -1)
#     mask = tf.where(mask <= triangle, 1, 0)

#     return tf.where(mask == 1, logits, -np.Infinity)


def use_mask(
    logits: tf.Tensor, mask: tf.Tensor
) -> tf.Tensor:  # assume that it has batch size in it
    mask = tf.cast(mask, tf.bool)  # type: ignore

    dim_diff = len(logits.shape) - len(mask.shape)
    for _ in range(dim_diff):
        mask = tf.expand_dims(mask, axis=1)

    return tf.where(mask, logits, -np.Infinity)


@dataclass(frozen=True)
class AttentionSpecs(BaseInitSpecs):
    is_causal: bool
    seq_len: int


class CrossAttention(keras.layers.Layer):
    def __init__(
        self,
        specs: AttentionSpecs,
        lora_specs: LoraSpecs,
        layer_specs: LayerSpecs,
    ):
        self.layer_specs = layer_specs
        self.lora_specs = lora_specs
        super(CrossAttention, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        self.specs = specs

        self.q: Extractor
        self.k: Extractor
        self.v: Extractor
        self.out: OutProj
        self.causal_mask: tf.Tensor
        self.casual: Callable[[tf.Tensor], tf.Tensor]

    def __get_causal_mask(self, seq_len: int) -> tf.Tensor:
        # mask = tf.linalg.band_part(tf.ones_like(logits, tf.uint8), -1, 0)
        mask: tf.Tensor = tf.ones((seq_len,), dtype=tf.int32)
        mask = tf.cumsum(mask)
        mask = tf.expand_dims(mask, 0)
        mask = tf.repeat(mask, seq_len, axis=0)

        triangle = tf.range(seq_len, dtype=tf.int32)
        triangle = tf.add(triangle, 1)
        triangle = tf.expand_dims(triangle, -1)

        mask = tf.where(mask <= triangle, 1, 0)
        return mask

    def __build_initializer_child(self, t: type, sufix: str, shape) -> BaseInitializer:
        child: BaseInitializer = t(
            self.specs,
            self.lora_specs,
            replace(self.layer_specs, name=f"{self.layer_specs.name}_{sufix}"),
        )
        child.build(shape)
        return child

    def build(self, input_shape):
        super().build(input_shape)

        self.q: Extractor = self.__build_initializer_child(Extractor, "q", input_shape)
        self.k: Extractor = self.__build_initializer_child(Extractor, "k", input_shape)
        self.v: Extractor = self.__build_initializer_child(Extractor, "v", input_shape)
        self.out: OutProj = self.__build_initializer_child(OutProj, "out", input_shape)

        self.causal_mask = self.__get_causal_mask(self.specs.seq_len)
        if self.specs.is_causal:
            self.casual = lambda logits: tf.where(
                self.causal_mask == 1, logits, -np.Infinity
            )
        else:
            self.casual = lambda logits: logits

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        *args,
        **kwargs,
    ) -> tf.Tensor:
        encoded, x, mask, condition = inputs  # bs*s*e, bs*s*e, bs*s, bs*1
        q = self.q([x, condition])  # bs*h*s*q
        k = self.k([encoded, condition])  # bs*h*s*q
        v = self.v([encoded, condition])  # bs*h*s*q
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # bs*s*h*q
        out = self.out([v, condition])  # bs*h*s*e

        logits = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))  # bs*h*s*s
        logits = logits / tf.math.sqrt(tf.cast(k.shape[-1], self.dtype))  # bs*h*s*s
        logits = self.casual(logits)  # bs*h*s*s
        logits = use_mask(logits, mask)  # bs*h*s*s

        prob = tf.nn.softmax(logits, axis=-1)  # bs*h*s*s
        prob = tf.transpose(prob, perm=[0, 1, 3, 2])  # bs*h*s*s
        prob = tf.expand_dims(prob, axis=-1)  # bs*h*s*s*1

        out = tf.expand_dims(out, axis=-2)  # bs*h*s*1*e
        out = tf.matmul(prob, out)  # bs*h*s*s*e
        out = tf.transpose(out, perm=[0, 1, 3, 2, 4])  # bs*h*s*s*e
        out = tf.reduce_sum(out, axis=-2)  # bs*h*s*e
        out = tf.transpose(out, perm=[0, 2, 1, 3])  # bs*s*h*e
        out = tf.reduce_sum(out, axis=-2)  # bs*s*e
        return out
