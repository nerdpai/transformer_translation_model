from keras.callbacks import EarlyStopping

from module_4_5.preparations.train_components.history import History
from module_4_5.preparations.train_components.ranged_decay import RangedDecay
from module_4_5.preparations.train_components.bin_crossentropy_acc import (
    BinaryCrossentopyAccuracy,
)
from module_4_5.preparations.train_components.bool_crossentropy_acc import (
    BoolCrossentopyAccuracy,
)


class TrainComponents:
    def __init__(
        self,
        history: History,
        decay: RangedDecay,
        early_stopping: EarlyStopping,
        binary_accuracy_metric: BinaryCrossentopyAccuracy,
        bool_accuracy_metric: BoolCrossentopyAccuracy,
    ):
        self.history = history
        self.decay = decay
        self.early_stopping = early_stopping
        self.binary_accuracy_metric = binary_accuracy_metric
        self.bool_accuracy_metric = bool_accuracy_metric


def execute(
    init_lr: float,
    final_lr: float,
    patience_in_epochs: int,
    patience_monitor: str,
) -> TrainComponents:
    history = History()
    decay = RangedDecay(init_lr, final_lr)
    early_stopping = EarlyStopping(
        monitor=patience_monitor, patience=patience_in_epochs
    )
    binary_accuracy_metric = BinaryCrossentopyAccuracy()
    bool_accuracy_metric = BoolCrossentopyAccuracy()

    return TrainComponents(
        history, decay, early_stopping, binary_accuracy_metric, bool_accuracy_metric
    )
