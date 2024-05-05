import tensorflow as tf
import numpy as np


def create_context_pairs(sequence: tf.Tensor, window: int) -> tf.Tensor:
    range_tensor = tf.range(len(sequence))

    indexes = tf.expand_dims(range_tensor, axis=-1) + tf.range(-window, window + 1)
    mask = tf.logical_and(indexes >= 0, indexes < tf.shape(sequence)[0])
    indexes = tf.where(mask, indexes, 0)

    context = tf.gather(sequence, indexes, validate_indices=False)
    context = tf.where(mask, context, -1)
    context = tf.concat([context[:, :window], context[:, window + 1 :]], axis=1)

    target = tf.repeat(sequence, repeats=2 * window)
    context = tf.reshape(context, [-1])

    r = tf.stack([target, context], axis=-1)
    r = tf.boolean_mask(r, tf.not_equal(r[:, 1], -1))
    return r


def filter_rows(pairs: tf.Tensor, sampling_table: np.ndarray) -> tf.Tensor:
    sampling_table_t = tf.convert_to_tensor(sampling_table, dtype=tf.float32)
    first_col = pairs[:, 0]
    random_values = tf.random.uniform(shape=tf.shape(first_col), minval=0, maxval=1)

    selected_logits = tf.gather(sampling_table_t, first_col)
    row_mask = selected_logits > random_values
    saved_rows = tf.boolean_mask(pairs, row_mask)

    return saved_rows
