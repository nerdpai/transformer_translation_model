import math
import numpy as np
from keras.utils import Sequence
from typing import Tuple


from preparations.dataset_components.h5_dset import H5Dset


class NeighbourGenerator(Sequence):

    def __init__(
        self,
        h5_dset: H5Dset,
        batch_size: int,
        vocab_size: int,
        shuffle: bool = True,
        shuffle_overlap: float = 1.0,
    ):
        self.h5_dset = h5_dset
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.shuffle_overlap = shuffle_overlap

        self.__len = len(self.h5_dset)
        self.__batch_len = math.ceil(self.__len / batch_size)
        self.__shuffle_batch = int(self.__len * self.shuffle_overlap)
        self.on_epoch_end()

    def __prepare_to_train(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(batch)
        train = batch[:, 0]
        y = batch[:, 1:]
        lables = np.zeros((batch_size, self.vocab_size))

        lables[np.arange(batch_size).reshape(batch_size, 1), y] = 1
        lables = lables.astype(np.float16)

        return train, lables

    def on_epoch_end(self):
        if self.shuffle:
            self.h5_dset.shuffle(self.__shuffle_batch)

    def __len__(self):
        return self.__batch_len

    def __getitem__(self, index):
        s = index * self.batch_size
        e = s + self.batch_size
        batch = self.h5_dset.get_batch(s, e)
        return self.__prepare_to_train(batch)
