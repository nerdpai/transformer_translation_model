import datasets
import numpy as np
import tensorflow._api.v2.v2 as tf
from tqdm import tqdm


class SamplingTables:
    def __init__(self, default: np.ndarray, negative: np.ndarray) -> None:
        self.positive: np.ndarray = default
        self.negative: np.ndarray = negative


def create_sampling_tables(
    dataset: datasets.Dataset,
    column_with_seqiences: str,
    batch_size: int,
    vocab_size: int,
    t: float,
) -> SamplingTables:
    tokens_appearance = tf.zeros(vocab_size, dtype=tf.int64)
    for i in tqdm(range(0, len(dataset), batch_size), desc="Sample table creation"):
        batch = dataset[i : i + batch_size]
        tokens = np.concatenate(
            batch[column_with_seqiences], dtype=np.int32
        )  # ignore linter error
        tokens = tf.convert_to_tensor(tokens, dtype=tf.int32)
        unique, _, counts = tf.unique_with_counts(tokens, out_idx=tf.int64)
        indices = tf.expand_dims(unique, axis=-1)
        tokens_appearance = tf.tensor_scatter_nd_add(tokens_appearance, indices, counts)

    total_tokens = tf.reduce_sum(tokens_appearance)

    sqrt_token_app = tf.sqrt(tf.cast(tokens_appearance, dtype=tf.float64))
    negative_sampling = sqrt_token_app / tf.reduce_sum(sqrt_token_app)

    frequency = tf.cast(tokens_appearance, dtype=tf.float64) / tf.cast(
        total_tokens, dtype=tf.float64
    )  # type: ignore
    weight_frequency: tf.Tensor = tf.where(frequency == 0, 0, t / (frequency + 1e-8))
    positive_sampling = weight_frequency + tf.sqrt(weight_frequency)

    return SamplingTables(positive_sampling.numpy(), negative_sampling.numpy())
