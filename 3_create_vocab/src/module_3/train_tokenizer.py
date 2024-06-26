import datasets
import pandas as pd
from typing import Iterator
from pathlib import Path
from tokenizers import (
    models,
    Tokenizer,
    trainers,
)


def __nan_remove(texts: list[str]) -> list[str]:
    df = pd.DataFrame(texts, columns=["data"])
    df = df[df["data"].notna()]
    return df["data"].tolist()


def data_generator(
    dataset: datasets.Dataset, batch_size: int, content_column: str
) -> Iterator[list[str]]:
    for i in range(0, len(dataset), batch_size):
        content: list[str] = dataset[i : i + batch_size][content_column]

        content = __nan_remove(content)

        yield content


def train_tokenizer(
    tokenizer: Tokenizer,
    trainer: trainers.BpeTrainer,
    dataset: datasets.Dataset,
    batch_size: int,
    content_column: str,
) -> Tokenizer:
    tokenizer.train_from_iterator(
        data_generator(dataset, batch_size, content_column),
        trainer=trainer,
        length=len(dataset),
    )
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = save_dir / "tokenizer.json"

    model: models.Model = tokenizer.model
    model.save(str(save_dir), "")
    tokenizer.save(str(tokenizer_path))


def execute(
    tokenizer: Tokenizer,
    trainer: trainers.BpeTrainer,
    dataset: datasets.Dataset,
    batch_size: int,
    save_dir: Path,
    content_column: str,
) -> None:
    tokenizer = train_tokenizer(tokenizer, trainer, dataset, batch_size, content_column)
    save_tokenizer(tokenizer, save_dir)
