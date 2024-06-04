import shutil
import h5py
import numpy as np
import pandas as pd
from datasets import Dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from pathlib import Path

from preparations.dataset_components.h5_dset import H5Dset

__DTYPE = np.int32


def __nan_remove(texts: list[str]) -> list[str]:
    df = pd.DataFrame(texts, columns=["data"])
    df = df[df["data"].notna()]
    return df["data"].tolist()


def __prepare_place(dset_path: Path, dset_name: str, window_size: int) -> None:
    with h5py.File(dset_path, "w") as h5f:
        h5f.create_dataset(
            dset_name,
            data=np.empty((0, window_size * 2 + 1)),
            compression=0,
            chunks=True,
            maxshape=(None, None),
            dtype=__DTYPE,
        )


def __transformation_loop(
    dataset: Dataset,
    content_column: str,
    tokenizer: Tokenizer,
    batch_size: int,
    window_size: int,
    dset_path: Path,
) -> None:
    with h5py.File(dset_path, "a") as h5f:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Data transformation"):
            data = dataset[i : i + batch_size]
            texts: list[str] = data[content_column]

            texts = __nan_remove(texts)
            if len(texts) == 0:
                continue

            fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            tokenized: np.ndarray = fast_tokenizer(texts, add_special_tokens=True, return_tensors="np")["input_ids"]  # type: ignore

            samples = __create_neighbour_samples(tokenized, window_size)

            __save_samples(h5f, content_column, samples)


def __create_neighbour_samples(sequences: np.ndarray, window_size: int) -> np.ndarray:
    window_shape = window_size * 2 + 1
    samples = np.zeros((0, window_shape), dtype=__DTYPE)
    for i in range(len(sequences)):
        seq = sequences[i]
        if len(seq) < window_shape:
            continue

        seq = np.array(seq, dtype=__DTYPE)
        sample = np.lib.stride_tricks.sliding_window_view(
            seq, window_shape=window_shape
        )
        sample = np.array(sample)
        sample[:, [0, window_size]] = sample[:, [window_size, 0]]

        samples = np.concatenate((samples, sample), axis=0, dtype=__DTYPE)

    return samples


def __save_samples(h5f: h5py.File, dset_name: str, samples: np.ndarray) -> None:
    dset: h5py.Dataset = h5f[dset_name]  # type: ignore
    dset.resize((dset.shape[0] + samples.shape[0]), axis=0)
    dset[-samples.shape[0] :, :] = samples


def __is_prepared(save_dir: Path) -> bool:
    if save_dir.exists():
        user_input = input(
            f"Dataset {save_dir.parent.name + '/' + save_dir.name} already transformed. Do you want to rewrite it? (y/n) [n]: "
        )
        if user_input.lower() != "y":
            print("Stopping transformation.")
            return True

    return False


def transform(
    dataset: Dataset,
    content_column: str,
    tokenizer: Tokenizer,
    batch_size: int,
    window_size: int,
    cache_dir: Path,
) -> H5Dset:

    dset_path = cache_dir / "dset.h5"
    r = H5Dset(dset_path, content_column)

    is_place_good = __is_prepared(cache_dir)
    if is_place_good:
        return r
    else:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    __prepare_place(dset_path, content_column, window_size)
    __transformation_loop(
        dataset, content_column, tokenizer, batch_size, window_size, dset_path
    )

    return r
