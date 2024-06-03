import h5py
import datasets
import numpy as np
from pathlib import Path
from tqdm import tqdm


import module_4.generator.tables.skip_grams as skip_grams
from module_4.generator.tables.sampling_tables import SamplingTables

__DTYPE = np.int32


def save_skip_sample(
    sequence: np.ndarray,
    h5f: h5py.File,
    window_size: int,
    skip_column_name: str,
    num_ns: int,
    sampl_tables: SamplingTables,
) -> None:
    save_place: h5py.Dataset = h5f[skip_column_name]  # type: ignore
    sample = skip_grams.skip_grams(sequence, window_size, num_ns, sampl_tables)

    save_place.resize((save_place.shape[0] + sample.shape[0]), axis=0)
    save_place[-sample.shape[0] :, :] = sample


def generate_skip_dataset(
    tokenized_dataset: datasets.Dataset,
    column_with_tokens: str,
    batch_size: int,
    window_size: int,
    num_ns: int,
    sampl_tables: SamplingTables,
    skip_dataset_path: Path,
    skip_column_name: str,
    start: int = 0,
    end: int = -1,
) -> None:

    if end == -1:
        end = len(tokenized_dataset)

    with h5py.File(skip_dataset_path, "w") as h5f:
        h5f.create_dataset(
            skip_column_name,
            data=np.empty((0, 2 + num_ns)),
            compression=0,
            chunks=True,
            maxshape=(None, None),
            dtype=__DTYPE,
        )

    with h5py.File(skip_dataset_path, "a") as h5f:
        for i in tqdm(range(start, end, batch_size), desc="Skip creation"):
            s = i
            e = i + batch_size
            e = min(e, end)
            dicted_data = tokenized_dataset[s:e]
            data: np.ndarray = np.concatenate(dicted_data[column_with_tokens])
            save_skip_sample(
                data, h5f, window_size, skip_column_name, num_ns, sampl_tables
            )
