import unittest
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass


from module_5.model.layers.transformer.cross_attention import (
    CrossAttention,
    AttentionSpecs,
    LoraSpecs,
    LayerSpecs,
)

# pylint: disable=e0401
from trainable_test import TrainableTest


@dataclass
class Constants:
    emb_dim: int = 3
    heads_num: int = 3
    is_causal: bool = False
    loras_num: int = 1
    which_lora: int = 0
    rank: int = 8
    alpha: int = 32
    learning_rate: float = 1e-1
    epochs_num: int = 25
    verbose: int = 0
    repeat_num: int = 10


class TestAttentions(unittest.TestCase, TrainableTest):
    def prepare_data(self) -> None:
        data = tf.constant(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=tf.float32,
        )
        self.data = tf.reshape(data, (1, 3, 3))
        self.target = tf.multiply(self.data, 10)
        mask = tf.constant(
            [1, 1, 0],
            dtype=tf.int8,
        )
        self.mask = tf.reshape(mask, (1, 3))
        cond = tf.constant(
            [self.constants.which_lora],
            dtype=tf.int32,
        )
        self.cond = tf.reshape(cond, (1, 1))

    def prepare_layers(self) -> None:
        atten_specs = AttentionSpecs(
            emb_dim=self.constants.emb_dim,
            heads_num=self.constants.heads_num,
            is_causal=self.constants.is_causal,
            seq_len=self.data.shape[1],
        )
        lora_specs = LoraSpecs(
            rank=self.constants.rank,
            alpha=self.constants.alpha,
            loras_num=self.constants.loras_num,
        )
        layer_specs = LayerSpecs(
            trainable=True,
            name="cross_attention",
            dtype=tf.float32,
        )
        self.cross_attention = CrossAttention(atten_specs, lora_specs, layer_specs)

    def prepare_model(self) -> None:
        sequence_size = self.data.shape[1]
        emb_dim = self.constants.emb_dim

        key_inp = keras.Input(
            shape=(sequence_size, emb_dim),
            dtype=tf.float32,
        )
        query_inp = keras.Input(
            shape=(sequence_size, emb_dim),
            dtype=tf.float32,
        )
        mask_inp = keras.Input(
            shape=(sequence_size,),
            dtype=tf.int8,
        )
        cond_inp = keras.Input(
            shape=(1,),
            dtype=tf.int32,
        )

        self.model = keras.Model(
            inputs=[key_inp, query_inp, mask_inp, cond_inp],
            outputs=self.cross_attention([key_inp, query_inp, mask_inp, cond_inp]),
        )
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.constants.learning_rate),
            loss=tf.losses.MeanSquaredError(),
        )

    def train(self) -> None:
        r = self.constants.repeat_num
        key = self._get_repeat_tensor(self.data, r)
        query = self._get_repeat_tensor(self.data, r)
        mask = self._get_repeat_tensor(self.mask, r)
        cond = self._get_repeat_tensor(self.cond, r)
        target = self._get_repeat_tensor(self.target, r)

        self.model.fit(
            x=[key, query, mask, cond],
            y=target,
            epochs=self.constants.epochs_num,
            verbose=self.constants.verbose,  # type: ignore
        )

    def evaluate(self) -> float:
        return self.model.evaluate(
            x=[self.data, self.data, self.mask, self.cond],
            y=self.target,
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

    def test_cross_attention(self) -> None:
        self.assertTrue(
            self.before_train > self.after_train,
            self._before_after_text(
                "cross_attention",
                self.before_train,
                self.after_train,
            ),
        )


if __name__ == "__main__":
    unittest.main()
