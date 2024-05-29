import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers.schedules import LearningRateSchedule  # type: ignore


class History(Callback):
    def __init__(self):
        super().__init__()
        self.batch_history = {
            "batch_loss": [],
            "batch_categorical_accuracy": [],
            "epoch_loss": [],
            "epoch_categorical_accuracy": [],
        }

    def __log(self, logs: dict, version: str):
        self.batch_history[f"{version}_loss"].append(logs["loss"])
        self.batch_history[f"{version}_categorical_accuracy"].append(
            logs["categorical_accuracy"]
        )

    def on_train_batch_end(self, batch, logs={}):
        self.__log(logs, "batch")

    def on_epoch_end(self, epoch, logs={}):
        self.__log(logs, "epoch")

    def save_history(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)

        for version in ["batch", "epoch"]:
            fig: Figure
            ax1: Axes

            fig, ax1 = plt.subplots()

            color = "red"
            ax1.set_xlabel(version)
            ax1.set_ylabel("loss", color=color)
            ax1.plot(self.batch_history[f"{version}_loss"], color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            ax2: Axes = ax1.twinx()  # type: ignore

            color = "green"
            ax2.set_ylabel("accuracy", color=color)
            ax2.plot(self.batch_history[f"{version}_categorical_accuracy"], color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            fig.tight_layout(rect=(0, 0, 1, 0.96))
            plt.title("model training")
            plt.savefig(output_path / f"{version}_history.png")
            plt.close()


class RangedDecay(LearningRateSchedule):
    def __init__(self, initial_lr: float, final_lr: float, steps_num: int):
        super().__init__()
        self.initial_lr: float = initial_lr
        self.decay: float = (initial_lr - final_lr) / steps_num
        self.final_lr: float = final_lr

    def __call__(self, step):
        lr = self.initial_lr - self.decay * tf.cast(step, tf.float32)
        return tf.maximum(lr, tf.constant(self.final_lr, dtype=tf.float32))


def execute(
    init_lr: float,
    final_lr: float,
    epochs_num: int,
    parts_per_epoch: int,
    part_size: int,
    samples_per_line: int,
    train_batch_size: int,
    patience_for_epoch: int,
) -> Tuple[History, RangedDecay, EarlyStopping]:
    history = History()
    steps_num: int = (
        epochs_num * parts_per_epoch * part_size * samples_per_line // train_batch_size
    )
    decay = RangedDecay(init_lr, final_lr, steps_num)
    early_stopping = EarlyStopping(monitor="loss", patience=patience_for_epoch)
    return history, decay, early_stopping
