import unittest
import numpy as np
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass


from module_5.model.layers.lora import Lora, LoraSpecs, LayerSpecs

# pylint: disable=e0401
from trainable_test import TrainableTest


@dataclass(frozen=True)
class Constants:
    loras_num: int = 3
    rank: int = 8
    alpha: int = 32
    epochs_num: int = 25
    batch_size: int = 1
    verbose: int = 0
    learning_rate: float = 5e-2
    epsilon: float = 5e-2


class TestLora(unittest.TestCase, TrainableTest):
    def prepare_data(self) -> None:
        self.data_list = list(range(-1, self.constants.loras_num))
        data = tf.convert_to_tensor(self.data_list, dtype=tf.int32)
        self.data = tf.reshape(data, (len(self.data_list), 1))

    def prepare_layers(self) -> None:
        self.lora = Lora(
            lora_specs=LoraSpecs(
                self.constants.rank,
                self.constants.alpha,
                self.constants.loras_num,
            ),
            shape=tf.TensorShape(dims=(1, 1)),
            layer_specs=LayerSpecs(dtype=tf.float32, name="lora"),
        )

    def prepare_model(self) -> None:
        lora_input = keras.Input(shape=(1,), dtype=tf.int32)

        self.model = keras.Model(inputs=lora_input, outputs=self.lora(lora_input))

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.constants.learning_rate),
            loss=keras.losses.MeanSquaredError(),
        )

    def train(self) -> None:
        self.model.fit(
            self.data,
            self.data,
            epochs=self.constants.epochs_num,
            batch_size=self.constants.batch_size,
            verbose=self.constants.verbose,  # type: ignore
        )

    def evaluate(self) -> None:
        default_target = self.data[:1]
        loras_target = self.data[1:]

        default: np.ndarray = self.model.predict(
            default_target,
            batch_size=self.constants.batch_size,
            verbose=self.constants.verbose,  # type: ignore
        )
        loras: np.ndarray = self.model.predict(
            loras_target,
            batch_size=self.constants.batch_size,
            verbose=self.constants.verbose,  # type: ignore
        )
        loras = np.squeeze(loras)

        self.default_check = default == 0.0
        self.max_dev = np.max(np.abs(loras - self.data_list[1:]))
        self.lora_check = self.max_dev < self.constants.epsilon

    def setUp(self) -> None:
        self.constants = Constants()

        self.prepare_data()
        self.prepare_layers()

        self.prepare_model()
        self.train()
        self.evaluate()

    def test_lora_layer(self) -> None:
        self.assertTrue(self.default_check, "the default lora matrix is not zero")

        self.assertTrue(
            self.lora_check,
            f"lora layers one output is larger then deviation {self.constants.epsilon} < {self.max_dev}",
        )


if __name__ == "__main__":
    unittest.main()
