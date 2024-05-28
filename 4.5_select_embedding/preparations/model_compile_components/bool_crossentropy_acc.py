from tensorflow.keras.metrics import Metric  # type: ignore
import tensorflow as tf


class BoolCrossentopyAccuracy(Metric):
    def __init__(self, name="bool_crossentopy_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.r: tf.Variable = tf.Variable(tf.constant(0.0), trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None) -> tf.Operation | None:
        y_true_bool = tf.cast(y_true, tf.bool)
        y_pred_bool = tf.where(y_pred >= 0.5, True, False)  # type: ignore
        good_predicted = tf.math.logical_not(
            tf.math.logical_xor(y_true_bool, y_pred_bool)
        )
        good_predicted = tf.cast(good_predicted, tf.float32)
        avg = tf.reduce_mean(good_predicted)
        self.r.assign(avg)

    def result(self):
        return self.r
