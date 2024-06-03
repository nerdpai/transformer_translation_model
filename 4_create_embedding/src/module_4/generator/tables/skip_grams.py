import tensorflow._api.v2.v2 as tf
import numpy as np

import module_4.generator.tables.sampling_tables as st
import module_4.generator.tables.positive_pairs as pp
import module_4.generator.tables.negative_sampling as ns


def skip_grams(
    sequence: np.ndarray,
    window_size: int,
    num_ns: int,
    sampl_tables: st.SamplingTables,
    epsilon: float = 1e-7,
) -> np.ndarray:
    sequence_t: tf.Tensor = tf.convert_to_tensor(sequence, dtype=tf.int32)

    pairs = pp.create_context_pairs(sequence_t, window_size)  # type: ignore
    pairs = pp.filter_rows(pairs, sampl_tables.positive)

    neg_samples = ns.negative_sampling(
        pairs[:, 1], num_ns, sampl_tables.negative, epsilon  # type: ignore
    )

    pairs = pairs.numpy()  # type: ignore
    neg_samples = neg_samples.numpy()  # type: ignore
    return np.concatenate((pairs, neg_samples), axis=1)
