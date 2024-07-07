from tensorflow._api.v2.v2 import keras
from pathlib import Path

from module_5.preparations.train_comps import TrainComponents, History
from module_5.preparations.generator import PreTrainGenerator


def train_model(
    model: keras.Model,
    generator: PreTrainGenerator,
    train_comps: TrainComponents,
    epochs_num: int,
) -> tuple[keras.Model, History]:

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=train_comps.decay),  # type: ignore
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalCrossentropy(),
        ],
    )

    model.fit(generator, epochs=epochs_num, callbacks=[train_comps.history, train_comps.early_stopping])  # type: ignore
    return model, train_comps.history


def execute(
    model: keras.Model,
    generator: PreTrainGenerator,
    train_comps: TrainComponents,
    epochs_num: int,
    history_dir: Path,
) -> keras.Model:

    model, history = train_model(model, generator, train_comps, epochs_num)
    history.save_history(history_dir)

    return model
