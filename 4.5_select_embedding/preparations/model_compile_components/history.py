import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from keras.callbacks import Callback
from pathlib import Path


class History(Callback):
    def __init__(self):
        super().__init__()
        self.batch_history: dict[str, list] = {}
        self.versions = ["batch", "epoch"]
        self.initialize = False

    def __init_history(self, logs: dict):
        for version in self.versions:
            for key in logs.keys():
                self.batch_history[f"{version}_{key}"] = []
        self.initialize = True

    def __log(self, logs: dict, version: str):
        if not self.initialize:
            self.__init_history(logs)

        for key in logs.keys():
            self.batch_history[f"{version}_{key}"].append(logs[key])

    def on_train_batch_end(self, batch, logs={}):
        self.__log(logs, "batch")

    def on_epoch_end(self, epoch, logs={}):
        self.__log(logs, "epoch")

    def save_history(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)

        for version in self.versions:
            name = version.capitalize()

            ax: Axes
            _, ax = plt.subplots(figsize=(12, 8), dpi=200)
            for key, values in self.batch_history.items():
                if key.startswith(version):
                    label = key.replace(f"{version}_", "")
                    ax.plot(values, label=label)
            ax.set_title(f"{name} History")
            ax.set_xlabel(name)
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.tick_params(axis="both", which="minor", labelsize=8)
            legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
            frame = legend.get_frame()
            frame.set_facecolor("white")
            frame.set_alpha(0.75)
            plt.savefig(output_path / f"{version}_history.png", bbox_inches="tight")
            plt.close()
