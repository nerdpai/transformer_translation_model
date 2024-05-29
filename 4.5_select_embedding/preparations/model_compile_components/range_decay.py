import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule  # type: ignore


class RangedDecay(LearningRateSchedule):
    def __init__(self, initial_lr: float, final_lr: float):
        super().__init__()
        self.initial_lr: float = initial_lr
        self.final_lr: float = final_lr
        self.decay = None  # type: ignore

    def set_steps_num(self, steps_num: int) -> None:
        self.decay: float = (self.initial_lr - self.final_lr) / steps_num

    def __call__(self, step):
        if self.decay is None:
            raise ValueError("steps_num is not set")

        lr = self.initial_lr - self.decay * tf.cast(step, tf.float32)
        return tf.maximum(lr, tf.constant(self.final_lr, dtype=tf.float32))
