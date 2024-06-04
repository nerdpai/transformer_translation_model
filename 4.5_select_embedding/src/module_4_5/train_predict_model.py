from tensorflow._api.v2.v2 import keras
from pathlib import Path

from preparations.compile_elements import CompileElements, History
from preparations.dataset_components.generator import NeighbourGenerator


def train_model(
    emb: keras.layers.Embedding,
    generator: NeighbourGenerator,
    comp_elements: CompileElements,
    epochs_num: int,
) -> tuple[keras.Model, History]:
    vocab_size = emb.input_dim

    emb.trainable = False
    model = keras.Sequential(
        [
            emb,
            keras.layers.Dense(vocab_size, activation=keras.activations.sigmoid),
        ]
    )

    comp_elements.decay.set_steps_num(len(generator) * epochs_num)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=comp_elements.decay),  # type: ignore
        loss=keras.losses.binary_crossentropy,
        metrics=[
            comp_elements.binary_accuracy_metric,
            comp_elements.bool_accuracy_metric,
        ],
    )

    model.fit(generator, epochs=epochs_num, callbacks=[comp_elements.history, comp_elements.early_stopping])  # type: ignore
    return model, comp_elements.history


def execute(
    emb: keras.layers.Embedding,
    generator: NeighbourGenerator,
    comp_elements: CompileElements,
    epochs_num: int,
    history_dir: Path,
) -> keras.Model:

    model, history = train_model(emb, generator, comp_elements, epochs_num)
    history.save_history(history_dir)

    return model
