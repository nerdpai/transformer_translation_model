from tensorflow.keras import Model
from keras import optimizers, metrics, losses, utils, Sequential
from tokenizers import Tokenizer
from pathlib import Path

import generator.skip_grams_gen as gen
from word2vec import Word2Vec, EMBEDDING_LAYER_NAME


def train_word2vec(
    generator: utils.Sequence,
    vocab_size: int,
    embedding_dim: int,
    epochs_num: int,
    learning_rate: float,
    tokenizer: Tokenizer,
    pad_token: str,
) -> Model:
    word2vec_model: Word2Vec = Word2Vec(vocab_size, embedding_dim)  # type: ignore
    word2vec_model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=metrics.CategoricalAccuracy(),
    )

    word2vec_model.fit(generator, epochs=epochs_num)  # type: ignore
    word2vec_model.zero_padding(tokenizer.token_to_id(pad_token))
    return word2vec_model


def save_embedding(model: Model, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    for layer in model.layers:
        if layer.name == EMBEDDING_LAYER_NAME:
            embedding_model = Sequential([layer])
            embedding_model.save(str(output_path / "model.h5"))
            break


def execute(
    generator: gen.SkipGramsGenerator,
    tokenizer: Tokenizer,
    embedding_dim: int,
    epochs_num: int,
    learning_rate: float,
    output_path: Path,
    pad_token: str,
) -> None:

    model = train_word2vec(
        generator,
        tokenizer.get_vocab_size(),
        embedding_dim,
        epochs_num,
        learning_rate,
        tokenizer,
        pad_token,
    )
    save_embedding(model, output_path)
