import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule
from typing import Tuple


class History(Callback):
    def __init__(self):
        super().__init__()
        self.batch_history = {
            "batch_loss": [],
            "batch_categorical_accuracy": [],
            "epoch_loss": [],
            "epoch_categorical_accuracy": [],
        }

    def __log(self, logs, version: str):
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
            fig, ax1 = plt.subplots()

            color = "red"
            ax1.set_xlabel(version)
            ax1.set_ylabel("loss", color=color)
            ax1.plot(self.batch_history[f"{version}_loss"], color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            ax2 = ax1.twinx()

            color = "green"
            ax2.set_ylabel("accuracy", color=color)
            ax2.plot(self.batch_history[f"{version}_categorical_accuracy"], color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            fig.tight_layout(rect=(0, 0, 1, 0.96))
            plt.title("model training")
            plt.savefig(output_path / f"{version}_history.png")
            plt.close()


class RangedDecay(LearningRateSchedule):
    def __init__(self, initial_lr: float, final_lr: float, epochs_num: int):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay = (initial_lr - final_lr) / epochs_num

    def __call__(self, step):
        return self.initial_lr - self.decay * step


def execute(
    init_lr: float,
    final_lr: float,
    epochs_num: int,
    patience_for_epoch: int,
) -> Tuple[History, RangedDecay, EarlyStopping]:
    history = History()
    decay = RangedDecay(init_lr, final_lr, epochs_num)
    early_stopping = EarlyStopping(monitor="loss", patience=patience_for_epoch)
    return history, decay, early_stopping
