from tensorflow._api.v2.v2 import keras


from module_4.preparations.train_components.history import History

# pylint: disable=E0401, E0611
from module_4.preparations.train_components.ranged_decay import RangedDecay


def execute(
    init_lr: float,
    final_lr: float,
    epochs_num: int,
    parts_per_epoch: int,
    part_size: int,
    samples_per_line: int,
    train_batch_size: int,
    patience_for_epoch: int,
) -> tuple[History, RangedDecay, keras.callbacks.EarlyStopping]:
    history = History()
    steps_num: int = (
        epochs_num * parts_per_epoch * part_size * samples_per_line // train_batch_size
    )
    decay = RangedDecay(init_lr, final_lr, steps_num)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss", patience=patience_for_epoch
    )
    return history, decay, early_stopping
