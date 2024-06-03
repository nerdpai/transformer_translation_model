import datasets
import tokenizers
import numpy as np
import h5py
import math
import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from pathlib import Path
from tqdm import tqdm


import module_4.generator.tables.sampling_tables as st
import module_4.generator.datasets.tokenize as td
import module_4.generator.datasets.skip as sd


class SkipGenSpecs:
    def __init__(
        self,
        dataset: datasets.Dataset,
        column_with_text: str,
        tokenizer: tokenizers.Tokenizer,
        cache_dir: Path,
        batch_size: int = 32,
        window_size: int = 2,
        num_ns: int = 5,
        part_size: int = 2**12,
        parts_per_epoch: int = 10,
        part_interfere: int = 3,
        t: float = 10e-4,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.column_with_text = column_with_text
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_ns = num_ns
        self.part_size = part_size
        self.parts_per_epoch = parts_per_epoch
        self.t = t
        self.part_interfere = part_interfere
        self.shuffle = shuffle


class SkipGramsGenerator(keras.utils.Sequence):
    def __init__(
        self,
        specs: SkipGenSpecs,
    ):
        self.TOKENIZED_COL_NAME = "tokenized"
        self.SKIPPED_COL_NAME = "skip_grams"
        self.LABLES: np.ndarray = self.__get_lables(specs.num_ns)

        self.dataset = specs.dataset
        self.column_with_text = specs.column_with_text
        self.tokenizer = specs.tokenizer
        self.batch_size = specs.batch_size
        self.window_size = specs.window_size
        self.num_ns = specs.num_ns
        self.part_size = specs.part_size
        self.parts_per_epoch = specs.parts_per_epoch
        self.t = specs.t
        self.part_interfere = specs.part_interfere
        self.shuffle = specs.shuffle

        self.tokenized = specs.cache_dir / "tokenized"
        self.skipped = specs.cache_dir / "skipped.h5"

        td.tokenize_dataset(
            specs.dataset,
            specs.tokenizer,
            self.tokenized,
            specs.column_with_text,
            specs.part_size,
            self.TOKENIZED_COL_NAME,
        )

        self.sampl_tables: st.SamplingTables = st.create_sampling_tables(
            self.__get_cached_dataset(self.tokenized),
            self.TOKENIZED_COL_NAME,
            specs.part_size,
            specs.tokenizer.get_vocab_size(),
            specs.t,
        )

        self.cur_len: int = 0
        self.cur_part: int = 0
        self.part_num: int = math.ceil(len(specs.dataset) / specs.part_size)

        self.on_epoch_end()

    def __get_lables(self, num_ns: int) -> np.ndarray:
        ones = np.ones((1,))
        zeros = np.zeros((num_ns,))
        return np.concatenate([ones, zeros], axis=0)

    def __get_cached_dataset(self, cache_dir: Path) -> datasets.Dataset:
        dsets = []
        for file in cache_dir.glob("*.data"):
            dset = datasets.load_from_disk(str(file))
            dsets.append(dset)
        return datasets.concatenate_datasets(dsets)

    def __shuffle_skip(
        self, cached_skip: Path, column_name: str, parts_interfere: int
    ) -> None:
        with h5py.File(cached_skip, "a") as h5f:
            dset: h5py.Dataset = h5f[column_name]  # type: ignore
            num_of_samples = dset.shape[0]

            step: int = int((parts_interfere / self.parts_per_epoch) * num_of_samples)
            for i in tqdm(range(0, num_of_samples, step), desc="Shuffling"):
                start = i
                end = i + step

                data = np.array(dset[start:end], dtype=dset.dtype)
                data_tensor: tf.Tensor = tf.convert_to_tensor(data, dtype=dset.dtype)
                data_tensor = tf.random.shuffle(data_tensor)
                data = data_tensor.numpy()  # type: ignore
                dset[start:end] = data

    def __set_cur_len(self, cached_skip: Path, column_name: str) -> None:
        with h5py.File(cached_skip, "r") as h5f:
            dset: h5py.Dataset = h5f[column_name]  # type: ignore
            self.cur_len = dset.shape[0]

    def on_epoch_end(self):
        start = self.cur_part * self.part_size
        end = start + self.part_size * self.parts_per_epoch
        if end > len(self.dataset):
            end = len(self.dataset)

        tokenized_dataset = self.__get_cached_dataset(self.tokenized)
        sd.generate_skip_dataset(
            tokenized_dataset,
            self.TOKENIZED_COL_NAME,
            self.part_size,
            self.window_size,
            self.num_ns,
            self.sampl_tables,
            self.skipped,
            self.SKIPPED_COL_NAME,
            start,
            end,
        )

        self.__set_cur_len(self.skipped, self.SKIPPED_COL_NAME)

        if self.shuffle:
            self.__shuffle_skip(self.skipped, self.SKIPPED_COL_NAME, self.part_size)

        self.cur_part += self.parts_per_epoch
        if self.cur_part >= self.part_num:
            self.cur_part = 0

    def __len__(self):
        return math.ceil(self.cur_len / self.batch_size)

    def __getitem__(self, index):
        """
        Returns: (targets, contexts), labels
        """
        with h5py.File(self.skipped, "r") as h5f:
            dset: h5py.Dataset = h5f[self.SKIPPED_COL_NAME]  # type: ignore
            batch = dset[index * self.batch_size : (index + 1) * self.batch_size]
            target: np.ndarray = batch[:, 0]
            context = batch[:, 1:]
            labels = np.repeat(
                np.expand_dims(self.LABLES, axis=0), target.shape[0], axis=0
            )
            return (target, context), labels