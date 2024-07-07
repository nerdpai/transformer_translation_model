import tensorflow._api.v2.v2 as tf


from module_5.model.layers.transformer.attention_elmenets.base_initializer import (
    BaseInitializer,
)


class Extractor(BaseInitializer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        x, condition = inputs  # bs*s*e, bs*1
        weights, biases = self._prepare_call(condition)
        # h*q*e, 1*h*q
        result = tf.add(tf.einsum("hqe,bse->bshq", weights, x), biases)
        # bs*s*h*q
        result = tf.transpose(result, perm=[0, 2, 1, 3])
        # bs*h*s*q
        return result
