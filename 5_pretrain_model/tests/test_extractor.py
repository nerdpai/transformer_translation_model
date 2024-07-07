import unittest
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass


from module_5.model.layers.transformer.attention_elmenets.extractor import (
    Extractor,
)
from module_5.model.layers.transformer.attention_elmenets.base_initializer import (
    BaseInitSpecs,
    LayerSpecs,
)
from module_5.model.layers.lora import LoraSpecs

# pylint: disable=e0401
from trainable_test import TrainableTest


@dataclass(frozen=True)
class Constants:
    emb_dim: int = 1
    heads_num: int = 1
    lora_num: int = 1
    which_lora: int = 0
    rank: int = 8
    alpha: int = 32
    learning_rate: float = 1e-1
    epochs_num: int = 25
    verbose: int = 0
    repeat_num: int = 10


class TestExtractor(unittest.TestCase, TrainableTest):

    def prepare_data(self) -> None:
        data = tf.convert_to_tensor([1, 2, 3], dtype=tf.float32)
        self.data = tf.reshape(data, [1, 3, self.constants.emb_dim])
        target = tf.convert_to_tensor([4, 5, 6], dtype=tf.float32)
        self.target = tf.reshape(target, [1, 3, self.constants.emb_dim])
        cond = tf.convert_to_tensor([self.constants.which_lora], dtype=tf.int32)
        self.cond = tf.reshape(cond, [1, 1])

    def prepare_layers(self) -> None:
        base_specs = BaseInitSpecs(
            self.constants.emb_dim,
            self.constants.heads_num,
        )
        lora_specs = LoraSpecs(
            self.constants.rank,
            self.constants.alpha,
            self.constants.lora_num,
        )
        layer_specs = LayerSpecs(True, "extractor", tf.float32)
        self.extractor = Extractor(base_specs, lora_specs, layer_specs)

    def prepare_model(self) -> None:
        extr_inp = keras.Input(
            shape=(self.data.shape[1], self.constants.emb_dim),
            dtype=tf.float32,
        )
        cond_inp = keras.Input(shape=(1,), dtype=tf.int32)
        out = self.extractor([extr_inp, cond_inp])

        self.model = keras.Model(inputs=[extr_inp, cond_inp], outputs=out)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.constants.learning_rate),
            loss=keras.losses.MeanSquaredError(),
        )

    def evaluate(self) -> float:
        return self.model.evaluate(
            [
                self.data,
                self.cond,
            ],
            self.target,
            verbose=self.constants.verbose,  # type: ignore
        )  # type: ignore

    def train(self) -> None:
        r = self.constants.repeat_num
        data = self._get_repeat_tensor(self.data, r)
        target = self._get_repeat_tensor(self.target, r)
        cond = self._get_repeat_tensor(self.cond, r)

        self.model.fit(
            [data, cond],
            target,
            epochs=self.constants.epochs_num,
            verbose=self.constants.verbose,  # type: ignore
        )

    def setUp(self) -> None:
        self.constants = Constants()

        self.prepare_data()
        self.prepare_layers()

        self.prepare_model()
        self.before_train = self.evaluate()
        self.train()
        self.after_train = self.evaluate()

    def test_extractor(self) -> None:
        self.assertTrue(
            self.before_train > self.after_train,
            self._before_after_text(
                "extractor",
                self.before_train,
                self.after_train,
            ),
        )


if __name__ == "__main__":
    unittest.main()
