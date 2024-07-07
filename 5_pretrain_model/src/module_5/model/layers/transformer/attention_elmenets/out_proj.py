import tensorflow._api.v2.v2 as tf


from module_5.model.layers.transformer.attention_elmenets.base_initializer import (
    BaseInitializer,
)


class OutProj(BaseInitializer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        x, condition = inputs  # bs*s*h*q, bs*1
        weights, biases = self._prepare_call(condition)
        # h*q*e, 1*h*q
        result = tf.einsum("bshq,hqe->bshe", tf.add(x, biases), weights)
        # bs*s*h*e
        result = tf.transpose(result, perm=[0, 2, 1, 3])
        # bs*h*s*e
        return result
