from pathlib import Path
from typing import Optional, Callable
from datasets import Dataset
from tokenizers import Tokenizer

from module_4_5.preparations.dataset_components.fetch import get_train_test
from module_4_5.preparations.dataset_components.extract_subset import get_subset
from module_4_5.preparations.dataset_components.transform_data import transform
from module_4_5.preparations.dataset_components.generator import NeighbourGenerator
from module_4_5.preparations.dataset_components.h5_dset import H5Dset


class FetcherSpecs:
    def __init__(
        self,
        cache_dir: Path,
        content_column: str,
        langs: list[str],
        subset_of_train: float,
        subset_of_test: float,
        seed: Optional[int] = None,
    ):
        self.cache_dir = cache_dir
        self.content_column = content_column
        self.langs = langs
        self.subset_of_train = subset_of_train
        self.subset_of_test = subset_of_test
        self.seed = seed


class GeneratorSpecs:
    def __init__(
        self,
        cache_dir: Path,
        tokenizer: Tokenizer,
        transform_batch_size: int,
        train_batch_size: int,
        window_size: int,
        shuffle: bool,
        shuffle_overlap: float,
    ):
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.transform_batch_size = transform_batch_size
        self.train_batch_size = train_batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        self.shuffle_overlap = shuffle_overlap


def execute(
    fetch_specs: FetcherSpecs, gen_specs: GeneratorSpecs
) -> tuple[NeighbourGenerator, NeighbourGenerator]:
    train_dset, test_dset = get_train_test(
        fetch_specs.cache_dir,
        fetch_specs.content_column,
        fetch_specs.langs,
        fetch_specs.seed,
    )
    train_dset = get_subset(train_dset, fetch_specs.subset_of_train)
    test_dset = get_subset(test_dset, fetch_specs.subset_of_test)

    lamb_transform: Callable[[Dataset, str], H5Dset] = lambda dset, subdir: transform(
        dset,
        fetch_specs.content_column,
        gen_specs.tokenizer,
        gen_specs.transform_batch_size,
        gen_specs.window_size,
        gen_specs.cache_dir / subdir,
    )

    train_h5 = lamb_transform(train_dset, "train")
    test_h5 = lamb_transform(test_dset, "test")

    lamb_gen: Callable[[H5Dset, bool, float], NeighbourGenerator] = (
        lambda h5, shuffle, shuffle_overlap: NeighbourGenerator(
            h5,
            gen_specs.train_batch_size,
            gen_specs.tokenizer.get_vocab_size(),
            shuffle,
            shuffle_overlap,
        )
    )

    train_generator = lamb_gen(
        train_h5,
        gen_specs.shuffle,
        gen_specs.shuffle_overlap,
    )
    test_generator = lamb_gen(
        test_h5,
        False,
        0.0,
    )

    return train_generator, test_generator
