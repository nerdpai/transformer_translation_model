import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras


class BinaryCrossentopyAccuracy(keras.metrics.Metric):
    def __init__(self, name="binary_crossentropy_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.r: tf.Variable = tf.Variable(tf.constant(0.0), trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None) -> tf.Operation | None:  # type: ignore
        diff = tf.abs(y_true - y_pred)
        avg = tf.reduce_mean(diff)
        self.r.assign(1 - avg)

    def result(self):
        return self.r
