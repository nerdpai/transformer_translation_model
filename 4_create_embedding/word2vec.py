import numpy as np
import tensorflow as tf
from keras import layers

EMBEDDING_LAYER_NAME = "w2v_embedding"


# code from here: "https://www.tensorflow.org/text/tutorials/word2vec"
class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()

        self.target_embedding = layers.Embedding(
            vocab_size, embedding_dim, name=EMBEDDING_LAYER_NAME
        )

        self.context_embedding = layers.Embedding(
            vocab_size, embedding_dim, name="w2v_context"
        )

    def call(self, inputs, training=None, mask=None):
        target, context = inputs
        # (batch,), (batch, context)

        word_emb = self.target_embedding(target)
        # (batch, embed)

        context_emb = self.context_embedding(context)
        # (batch, context, embed)

        dots: tf.Tensor = tf.einsum("be,bce->bc", word_emb, context_emb)
        # (batch, context)
        return dots

    def zero_padding(self, index_of_pad: int):
        weights: np.ndarray = self.target_embedding.get_weights()[0]
        weights[index_of_pad] = np.zeros(weights.shape[1])
        self.target_embedding.set_weights([weights])
