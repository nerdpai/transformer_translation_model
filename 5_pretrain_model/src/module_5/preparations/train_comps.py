from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass


from module_5.preparations.train_components.history import History

# pylint: disable=E0401, E0611
from module_5.preparations.train_components.ranged_decay import RangedDecay


@dataclass(frozen=True)
class TrainComponents:
    history: History
    decay: RangedDecay
    early_stopping: keras.callbacks.EarlyStopping


def execute(
    init_lr: float,
    final_lr: float,
    steps_num: int,
    patience_in_epochs: int,
    patience_monitor: str,
) -> TrainComponents:
    history = History()
    decay = RangedDecay(init_lr, final_lr, steps_num)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=patience_monitor, patience=patience_in_epochs
    )
    return TrainComponents(history, decay, early_stopping)
