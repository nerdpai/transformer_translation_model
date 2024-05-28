from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets,
)
from pathlib import Path
from typing import Tuple, Optional


def concat_datasets(
    dsets: list[Dataset], text_column: str, content_column: str
) -> Dataset:
    dset = concatenate_datasets(dsets)
    dset = dset.rename_column(text_column, content_column)
    dset = dset.remove_columns(
        [col for col in dset.column_names if col != content_column]
    )
    return dset


def load_hub_datasets(
    cache_dir: Path,
    langs: list[str],
    name: str,
    split: str,
) -> list[Dataset]:
    dsets = []
    for lang in langs:
        dset: Dataset = load_dataset(
            name,
            f"{lang}",
            trust_remote_code=True,
            cache_dir=str(cache_dir),
        )  # type: ignore
        dsets.append(dset[split])
    return dsets


def get_paws_datast(
    cache_dir: Path,
    content_column: str,
    langs: list[str],
) -> Dataset:
    dsets = load_hub_datasets(cache_dir, langs, "paws-x", "train")
    return concat_datasets(dsets, "sentence1", content_column)


def get_lambada_dataset(
    cache_dir: Path,
    content_column: str,
    langs: list[str],
) -> Dataset:
    dsets = load_hub_datasets(cache_dir, langs, "EleutherAI/lambada_openai", "test")
    return concat_datasets(dsets, "text", content_column)


def get_train_test(
    cache_dir: Path,
    content_column: str,
    langs: list[str],
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    paws = get_paws_datast(cache_dir, content_column, langs)
    lambada = get_lambada_dataset(cache_dir, content_column, langs)

    if seed is not None:
        paws = paws.shuffle(seed)
        paws = paws.flatten_indices()

    return paws, lambada
