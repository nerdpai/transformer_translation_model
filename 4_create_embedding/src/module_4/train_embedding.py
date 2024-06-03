from tensorflow._api.v2.v2 import keras
from tokenizers import Tokenizer
from pathlib import Path


import module_4.generator.skip_grams_gen as gen
from module_4.word2vec import Word2Vec
from module_4.preparations.callbacks import History, RangedDecay


def train_word2vec(
    generator: keras.utils.Sequence,
    vocab_size: int,
    embedding_dim: int,
    epochs_num: int,
    tokenizer: Tokenizer,
    pad_token: str,
    callbacks: tuple[History, RangedDecay, keras.callbacks.EarlyStopping],
) -> tuple[Word2Vec, History]:
    history, decay, early_stop = callbacks

    word2vec_model: Word2Vec = Word2Vec(vocab_size, embedding_dim)  # type: ignore
    word2vec_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=decay),  # type: ignore
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=keras.metrics.CategoricalAccuracy(),
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
    callbacks: tuple[History, RangedDecay, keras.callbacks.EarlyStopping],
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
