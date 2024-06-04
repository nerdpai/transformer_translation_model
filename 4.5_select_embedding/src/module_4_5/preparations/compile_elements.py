from keras.callbacks import EarlyStopping

from preparations.model_compile_components.history import History
from preparations.model_compile_components.range_decay import RangedDecay
from preparations.model_compile_components.bin_crossentropy_acc import (
    BinaryCrossentopyAccuracy,
)
from preparations.model_compile_components.bool_crossentropy_acc import (
    BoolCrossentopyAccuracy,
)


class CompileElements:
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
    patience_in_epoch: int,
) -> CompileElements:
    history = History()
    decay = RangedDecay(init_lr, final_lr)
    early_stopping = EarlyStopping(monitor="loss", patience=patience_in_epoch)
    binary_accuracy_metric = BinaryCrossentopyAccuracy()
    bool_accuracy_metric = BoolCrossentopyAccuracy()

    return CompileElements(
        history, decay, early_stopping, binary_accuracy_metric, bool_accuracy_metric
    )
