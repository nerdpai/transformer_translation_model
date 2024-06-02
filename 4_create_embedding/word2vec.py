import numpy as np
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from pathlib import Path


# code from here: "https://www.tensorflow.org/text/tutorials/word2vec"
class Word2Vec(keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()
        __NAME = "w2v"

        self.target_embedding = keras.layers.Embedding(
            vocab_size, embedding_dim, name=f"{__NAME}_emb"
        )

        self.context_embedding = keras.layers.Embedding(
            vocab_size, embedding_dim, name=f"{__NAME}_context"
        )

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], training=None, mask=None):
        target, context = inputs
        # (batch,), (batch, context)

        word_emb: tf.Tensor = self.target_embedding(target)
        # (batch, embed)

        context_emb: tf.Tensor = self.context_embedding(context)
        # (batch, context, embed)

        dots: tf.Tensor = tf.einsum("be,bce->bc", word_emb, context_emb)
        # (batch, context)
        return dots

    def zero_padding(self, index_of_pad: int):
        weights: np.ndarray = self.target_embedding.get_weights()[0]
        weights[index_of_pad] = np.zeros(weights.shape[1])
        self.target_embedding.set_weights([weights])

    def save_embedding(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)

        embedding_model = keras.Sequential([self.target_embedding])
        embedding_model.save(str(output_path / "model.h5"))
