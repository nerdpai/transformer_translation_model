from datasets import Dataset, load_dataset, concatenate_datasets
from typing import Callable
from pathlib import Path


def get_hub_dataset(
    name: str,
    language_prefix: str,
    text_column: str,
    cache_dir: Path,
    content_column: str,
) -> Dataset:
    langs = ["en", "de", "fr"]
    l_dsets = []
    for lang in langs:
        dset: Dataset = load_dataset(
            name,
            f"{language_prefix}{lang}",
            trust_remote_code=True,
            cache_dir=str(cache_dir),
        )  # type: ignore
        l_dsets.append(dset["train"])
    dset = concatenate_datasets(l_dsets)
    dset = dset.rename_column(text_column, content_column)
    dset = dset.remove_columns(
        [col for col in dset.column_names if col != content_column]
    )
    return dset


def get_wiki_dataset(cache_dir: Path, content_column: str) -> Dataset:
    return get_hub_dataset("wikipedia", "20220301.", "text", cache_dir, content_column)


def get_allenAI_c4_dataset(cache_dir: Path, content_column: str) -> Dataset:
    return get_hub_dataset("allenai/c4", "", "text", cache_dir, content_column)


def get_CC_mined_dataset(
    cc_mined_dir: Path, cache_dir: Path, content_column: str
) -> Dataset:
    dset: Dataset = load_dataset(
        "csv",
        data_dir=str(cc_mined_dir),
        split="train",
        cache_dir=str(cache_dir),
    )  # type: ignore
    dset = dset.remove_columns(
        [col for col in dset.column_names if col != content_column]
    )
    return dset


def prepare_dataset(
    dataset_extractors: list[Callable[[Path, str], Dataset]],
    cache_dir: Path,
    content_column: str,
) -> Dataset:
    dsets = []
    for extractor in dataset_extractors:
        dataset = extractor(cache_dir, content_column)
        dsets.append(dataset)
    return concatenate_datasets(dsets)


def execute(cc_mined_dir: Path, cache_dir: Path, content_column: str) -> Dataset:
    cc_mined = lambda c_dir, content_col: get_CC_mined_dataset(
        cc_mined_dir, c_dir, content_col
    )
    dataset_extractors: list[Callable[[Path, str], Dataset]] = [
        get_wiki_dataset,
        get_allenAI_c4_dataset,
        cc_mined,
    ]
    return prepare_dataset(dataset_extractors, cache_dir, content_column)
