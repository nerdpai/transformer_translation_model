import unittest
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass

from module_5.model.layers.transformer.positional_emb import (
    PositionalEmbedding,
    LoraSpecs,
    LayerSpecs,
)

# pylint: disable=e0401
from trainable_test import TrainableTest


@dataclass(frozen=True)
class Constants:
    data: list[int]
    target: list[int]
    loras_num: int = 1
    rank: int = 8
    alpha: int = 32
    epochs_num: int = 25
    batch_size: int = 1
    verbose: int = 0
    learning_rate: float = 1e-2
    repeat_num: int = 10


@dataclass(frozen=True)
class CompData:
    without_lora: float
    with_lora: float


class TestPosEmb(unittest.TestCase, TrainableTest):
    def prepare_data(self) -> None:
        self.data = tf.convert_to_tensor([self.constants.data], dtype=tf.float32)
        self.target = tf.convert_to_tensor([self.constants.target], dtype=tf.float32)

        self.cond_without_lora = tf.convert_to_tensor([[-1]], dtype=tf.int32)
        self.cond_with_lora = tf.convert_to_tensor([[0]], dtype=tf.int32)

        self.shape: tf.TensorShape = self.data.shape

    def prepare_layers(self) -> None:
        self.lora_specs = LoraSpecs(
            self.constants.rank,
            self.constants.alpha,
            self.constants.loras_num,
        )
        self.layer_specs = LayerSpecs(dtype=tf.float32, name="pos_emb")
        self.pos_emb = PositionalEmbedding(
            self.shape, self.lora_specs, self.layer_specs
        )

    def prepare_model(self) -> None:
        cond_inp = keras.Input(shape=(1,), dtype=tf.int32)
        data_inp = keras.Input(shape=(self.shape[-1],), dtype=tf.float32)

        inputs = [data_inp, cond_inp]
        self.model = keras.Model(
            inputs=inputs,
            outputs=self.pos_emb(inputs),
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.constants.learning_rate),
            loss=keras.losses.MeanSquaredError(),
        )

    def train(self) -> None:
        repeats = self.constants.repeat_num
        epochs = self.constants.epochs_num
        verbose = self.constants.verbose

        data = tf.repeat(self.data, repeats, axis=0)
        target = tf.repeat(self.target, repeats, axis=0)
        cond_without_lora = tf.repeat(self.cond_without_lora, repeats, axis=0)
        cond_with_lora = tf.repeat(self.cond_with_lora, repeats, axis=0)

        self.model.fit([data, cond_without_lora], target, epochs=epochs, verbose=verbose)  # type: ignore
        self.model.fit([target, cond_with_lora], data, epochs=epochs, verbose=verbose)  # type: ignore

    def evaluate(self) -> CompData:
        verbose = self.constants.verbose

        pred_without_lora = self.model.evaluate(
            [self.data, self.cond_without_lora],
            self.target,
            verbose=verbose,  # type: ignore
        )
        pred_with_lora = self.model.evaluate(
            [self.target, self.cond_with_lora],
            self.data,
            verbose=verbose,  # type: ignore
        )
        return CompData(pred_without_lora, pred_with_lora)  # type: ignore

    def setUp(self) -> None:
        self.constants = Constants(data=[1, 2, 3], target=[3, 8, 12])

        self.prepare_data()
        self.prepare_layers()

        self.prepare_model()
        self.before_train = self.evaluate()
        self.train()
        self.after_train = self.evaluate()

    def test_pos_emb(self) -> None:
        self.assertTrue(
            self.before_train.with_lora > self.after_train.with_lora,
            self._before_after_text(
                "lora layer",
                self.before_train.with_lora,
                self.after_train.with_lora,
            ),
        )

        self.assertTrue(
            self.before_train.without_lora > self.after_train.without_lora,
            self._before_after_text(
                "positional embedding",
                self.before_train.without_lora,
                self.after_train.without_lora,
            ),
        )


if __name__ == "__main__":
    unittest.main()
