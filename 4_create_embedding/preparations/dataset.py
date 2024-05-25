from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    IterableDataset,
)
from typing import Callable, TypeAlias, Optional
from pathlib import Path


Extractor: TypeAlias = Callable[[Path, list[str], str], Dataset]


def concat_datasets(
    dsets: list[Dataset], text_column: str, content_column: str
) -> Dataset:
    dset = concatenate_datasets(dsets)
    dset = dset.rename_column(text_column, content_column)
    dset = dset.remove_columns(
        [col for col in dset.column_names if col != content_column]
    )
    return dset


def get_hub_datasets(
    name: str,
    language_prefix: str,
    cache_dir: Path,
    langs: list[str],
    streaming: bool = False,
) -> list[Dataset]:
    dsets = []
    for lang in langs:
        dset: Dataset = load_dataset(
            name,
            f"{language_prefix}{lang}",
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            streaming=streaming,
        )  # type: ignore
        dsets.append(dset["train"])
    return dsets


def load_batched_dataset(cache_dir: Path) -> Dataset:
    folders = [folder for folder in cache_dir.iterdir()]
    dsets: list[Dataset] = [load_from_disk(str(folder)) for folder in folders]  # type: ignore
    dset = concatenate_datasets(dsets)
    return dset


def save_only_lines(
    i_dsets: list[IterableDataset],
    save_only: list[int],
    batch_size: int,
    cache_dirs: list[Path],
) -> list[Dataset]:
    for i, i_dset in enumerate(i_dsets):
        generator = i_dset.iter(batch_size)
        for j in range(0, save_only[i], batch_size):
            batch_dset = Dataset.from_dict(next(generator))

            batch_dset.save_to_disk(str(cache_dirs[i] / f"{j}"))

    dsets = [load_batched_dataset(cache_dir) for cache_dir in cache_dirs]
    return dsets


def get_wiki_dataset(cache_dir: Path, langs: list[str], content_column: str) -> Dataset:
    dsets = get_hub_datasets("wikipedia", "20220301.", cache_dir, langs)
    return concat_datasets(dsets, "text", content_column)


def get_allenAI_c4_dataset(
    cache_dir: Path,
    langs: list[str],
    content_column: str,
    batch_size: int,
    c4_sizes: list[int],
) -> Dataset:
    streaming = True
    save_only = c4_sizes
    c4_dir = cache_dir / "my_c4"
    drive_dirs = [c4_dir / lang for lang in langs]
    dsets: list[Dataset] = []

    if not all([dir.exists() for dir in drive_dirs]):
        i_dsets: list[IterableDataset] = get_hub_datasets("allenai/c4", "", cache_dir, langs, streaming)  # type: ignore
        dsets = save_only_lines(i_dsets, save_only, batch_size, drive_dirs)
    else:
        dsets = [load_batched_dataset(dir) for dir in drive_dirs]

    for i, dset in enumerate(dsets):
        dset = dset.remove_columns("timestamp")
        dsets[i] = dset

    return concat_datasets(dsets, "text", content_column)


def get_CC_mined_dataset(
    cc_mined_dir: Path,
    cache_dir: Path,
    langs: list[str],
    content_column: str,
) -> Dataset:
    dsets = []
    for lang in langs:
        dir = cc_mined_dir / lang
        lset: Dataset = load_dataset(
            "csv",
            data_dir=str(dir),
            split="train",
            cache_dir=str(cache_dir),
        )  # type: ignore
        dsets.append(lset)

    dset = concatenate_datasets(dsets)
    dset = dset.remove_columns(
        [col for col in dset.column_names if col != content_column]
    )
    return dset


def prepare_dataset(
    dataset_extractors: list[Extractor],
    cache_dir: Path,
    langs: list[str],
    content_column: str,
) -> Dataset:
    dsets = []
    for extractor in dataset_extractors:
        dataset = extractor(cache_dir, langs, content_column)
        dsets.append(dataset)
    return concatenate_datasets(dsets)


def execute(
    cc_mined_dir: Path,
    cache_dir: Path,
    content_column: str,
    batch_size: int,
    langs: list[str],
    c4_size: list[int],
    seed: Optional[int] = None,
) -> Dataset:
    cc_mined = lambda c_dir, langs, content_col: get_CC_mined_dataset(
        cc_mined_dir, c_dir, langs, content_col
    )
    allen_c4 = lambda c_dir, langs, content_col: get_allenAI_c4_dataset(
        c_dir, langs, content_col, batch_size, c4_size
    )
    dataset_extractors: list[Extractor] = [
        get_wiki_dataset,
        allen_c4,
        cc_mined,
    ]

    dset = prepare_dataset(dataset_extractors, cache_dir, langs, content_column)
    if seed is not None:
        dset = dset.shuffle(seed=seed)
        dset = dset.flatten_indices()
    return dset
