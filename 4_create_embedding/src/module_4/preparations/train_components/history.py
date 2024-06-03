import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensorflow._api.v2.v2 import keras


class History(keras.callbacks.Callback):
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

    def on_train_batch_end(self, batch, logs=None):
        self.__log(logs, "batch")  # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        self.__log(logs, "epoch")  # type: ignore

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
