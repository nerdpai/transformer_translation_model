import h5py
import numpy as np
import tensorflow._api.v2.v2 as tf
from pathlib import Path
from tqdm import tqdm


class H5Dset:
    def __init__(self, h5_path: Path, dset_name: str):
        self.h5_path = h5_path
        self.dset_name = dset_name

    def shuffle(self, shuffle_batch: int):
        with h5py.File(self.h5_path, "a") as h5f:
            dset: h5py.Dataset = h5f[self.dset_name]  # type: ignore
            num_of_samples = dset.shape[0]

            step: int = shuffle_batch
            for i in tqdm(range(0, num_of_samples, step), desc="Shuffling"):
                start = i
                end = i + step

                data = np.array(dset[start:end], dtype=dset.dtype)
                data_tensor: tf.Tensor = tf.convert_to_tensor(data, dtype=dset.dtype)
                data_tensor = tf.random.shuffle(data_tensor)
                data = data_tensor.numpy()  # type: ignore
                dset[start:end] = data

    def get_batch(self, start: int, end: int) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            dset: h5py.Dataset = h5f[self.dset_name]  # type: ignore
            return dset[start:end]

    def __len__(self):
        with h5py.File(self.h5_path, "r") as h5f:
            dset: h5py.Dataset = h5f[self.dset_name]  # type: ignore
            return dset.shape[0]
