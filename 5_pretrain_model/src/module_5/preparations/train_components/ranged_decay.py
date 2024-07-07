import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras


# pylint: disable=C0115, W0223
class RangedDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr: float, final_lr: float, steps_num: int):
        super().__init__()
        self.initial_lr: float = initial_lr
        self.decay: float = (initial_lr - final_lr) / steps_num
        self.final_lr: float = final_lr

    def __call__(self, step):
        lr = self.initial_lr - self.decay * tf.cast(step, tf.float32)  # type: ignore
        return tf.maximum(lr, tf.constant(self.final_lr, dtype=tf.float32))
