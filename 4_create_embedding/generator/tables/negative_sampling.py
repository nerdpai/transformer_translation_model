import tensorflow as tf
import numpy as np


def negative_sampling(
    targets: tf.Tensor,
    num_ns: int,
    negative_sampling_table: np.ndarray,
    epsilon: float = 1e-7,
) -> tf.Tensor:
    sampling_table_t = tf.convert_to_tensor(negative_sampling_table, dtype=tf.float64)

    targets_len = targets.shape[0]
    # target_sum = tf.reduce_sum(sampling_table_t)
    # multiplier = len(negative_sampling_table) / target_sum
    multiplier = len(negative_sampling_table)

    sampling_table_t = sampling_table_t * multiplier
    sampling_table_t = sampling_table_t + epsilon
    sampling_table_t = tf.cumsum(sampling_table_t)

    random_tensor = tf.random.uniform(
        shape=(targets.shape[0] * num_ns,),  # type: ignore
        minval=0,
        maxval=tf.reduce_max(sampling_table_t),
        dtype=tf.float64,
    )

    indices = tf.searchsorted(sampling_table_t, random_tensor, side="right")
    targets = tf.repeat(targets, num_ns)

    indices = tf.where(indices == targets, indices + 1, indices)
    indices = tf.where(indices >= len(negative_sampling_table), 0, indices)
    indices = tf.reshape(indices, (targets_len, num_ns))

    return indices
