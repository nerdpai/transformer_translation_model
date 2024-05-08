from keras.callbacks import EarlyStopping
from keras import optimizers, metrics, losses, utils
from tokenizers import Tokenizer
from pathlib import Path
from typing import Tuple


import generator.skip_grams_gen as gen
from word2vec import Word2Vec
from preparations.callbacks import History, RangedDecay


def train_word2vec(
    generator: utils.Sequence,
    vocab_size: int,
    embedding_dim: int,
    epochs_num: int,
    tokenizer: Tokenizer,
    pad_token: str,
    callbacks: Tuple[History, RangedDecay, EarlyStopping],
) -> Tuple[Word2Vec, History]:
    history, decay, early_stop = callbacks

    word2vec_model: Word2Vec = Word2Vec(vocab_size, embedding_dim)  # type: ignore
    word2vec_model.compile(
        optimizer=optimizers.Adam(learning_rate=decay),  # type: ignore
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=metrics.CategoricalAccuracy(),
    )

    word2vec_model.fit(generator, epochs=epochs_num, callbacks=[history, early_stop])  # type: ignore
    word2vec_model.zero_padding(tokenizer.token_to_id(pad_token))
    return word2vec_model, history


def execute(
    generator: gen.SkipGramsGenerator,
    tokenizer: Tokenizer,
    embedding_dim: int,
    epochs_num: int,
    output_path: Path,
    pad_token: str,
    callbacks: Tuple[History, RangedDecay, EarlyStopping],
) -> None:

    model, history = train_word2vec(
        generator,
        tokenizer.get_vocab_size(),
        embedding_dim,
        epochs_num,
        tokenizer,
        pad_token,
        callbacks,
    )
    model.save_embedding(output_path)
    history.save_history(output_path)
