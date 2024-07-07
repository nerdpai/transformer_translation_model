import tensorflow._api.v2.v2 as tf


from module_5.model.layers.transformer.cross_attention import (
    CrossAttention,
    AttentionSpecs,
)


class SelfAttention(CrossAttention):

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        *args,
        **kwargs,
    ) -> tf.Tensor:
        x, mask, condition = inputs  # bs*s*e, bs*s, bs*1
        return super().call((x, x, mask, condition))
