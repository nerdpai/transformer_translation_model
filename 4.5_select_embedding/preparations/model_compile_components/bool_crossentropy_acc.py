import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras


class BoolCrossentopyAccuracy(keras.metrics.Metric):
    def __init__(self, name="bool_crossentopy_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.r: tf.Variable = tf.Variable(tf.constant(0.0), trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None) -> tf.Operation | None:  # type: ignore
        y_true_bool = tf.cast(y_true, tf.bool)
        y_pred_bool = tf.where(y_pred >= 0.5, True, False)
        good_predicted = tf.math.logical_not(
            tf.math.logical_xor(y_true_bool, y_pred_bool)
        )
        good_predicted = tf.cast(good_predicted, tf.float32)
        avg = tf.reduce_mean(good_predicted)
        self.r.assign(avg)

    def result(self):
        return self.r
