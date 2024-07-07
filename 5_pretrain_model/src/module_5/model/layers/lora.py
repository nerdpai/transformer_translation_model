import functools
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from typing import Callable
from dataclasses import dataclass, replace

from module_5.model.layers.lora_unit import LoraUnit, LoraUnitSpecs
from module_5.model.layers.layer_specs import LayerSpecs


@dataclass(frozen=True)
class LoraSpecs(LoraUnitSpecs):
    loras_num: int


class Lora(keras.layers.Layer):
    def __init__(
        self,
        lora_specs: LoraSpecs,
        shape: tf.TensorShape,
        layer_specs: LayerSpecs,
    ):
        self.lora_specs = lora_specs
        self.layer_specs = layer_specs
        super(Lora, self).__init__(
            layer_specs.trainable,
            layer_specs.name,
            layer_specs.dtype,
            layer_specs.dynamic,
        )

        self.shape = shape
        self.loras_num = lora_specs.loras_num

        self.zeros: tf.Tensor
        self.default_lambda: Callable[[], tf.Tensor]
        self.lora_units: list[LoraUnit]
        self.lora_branches: list[Callable[[], tf.Tensor]]

    def build(self, input_shape) -> None:
        super().build(input_shape)

        self.zeros = tf.zeros(shape=self.shape, dtype=self.dtype)
        self.default_lambda = lambda: self.zeros

        self.lora_units = [
            LoraUnit(
                self.shape,
                self.lora_specs,
                replace(self.layer_specs, name=f"{self.layer_specs.name}_{i}"),
            )
            for i in range(self.loras_num)
        ]
        for unit in self.lora_units:
            unit.build(input_shape)

        self.lora_branches = [
            functools.partial(layer, self.zeros) for layer in self.lora_units
        ]

        self.pred_fn_pairs = [
            (tf.equal(i, 0), self.lora_branches[i]) for i in range(self.loras_num)
        ]

    def call(
        self, inputs: tf.Tensor, *args, **kwargs
    ) -> tf.Tensor:  # should be the same lora for entire batch
        condition = inputs  # bs*1
        condition = tf.reduce_mean(condition, 0)  # 1*1
        condition = tf.squeeze(condition)  # 1

        lora_outputs = tf.case(
            pred_fn_pairs=self.pred_fn_pairs,
            default=self.default_lambda,
            exclusive=True,
        )

        return lora_outputs
