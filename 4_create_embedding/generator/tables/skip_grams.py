import tensorflow as tf
import numpy as np

import generator.tables.sampling_tables as st
import generator.tables.positive_pairs as pp
import generator.tables.negative_sampling as ns


def skip_grams(
    sequence: np.ndarray,
    window_size: int,
    num_ns: int,
    sampl_tables: st.SamplingTables,
    epsilon: float = 1e-7,
) -> np.ndarray:
    sequence_t = tf.convert_to_tensor(sequence, dtype=tf.int32)

    pairs = pp.create_context_pairs(sequence_t, window_size)
    pairs = pp.filter_rows(pairs, sampl_tables.positive)

    neg_samples = ns.negative_sampling(
        pairs[:, 1], num_ns, sampl_tables.negative, epsilon
    )

    pairs = pairs.numpy()
    neg_samples = neg_samples.numpy()
    return np.concatenate((pairs, neg_samples), axis=1)
