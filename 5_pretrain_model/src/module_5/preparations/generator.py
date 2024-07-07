# pylint: disable=c0200

import math
import numpy as np
import tensorflow._api.v2.v2 as tf
from datasets import Dataset
from tokenizers import Tokenizer, Encoding
from transformers import PreTrainedTokenizerFast, BatchEncoding
from tensorflow._api.v2.v2 import keras
from typing import Union


def _determine_avg_seq_len(
    dataset: Dataset,
    column_with_text: str,
    tokenizer: Tokenizer,
    max_samples: int,
) -> int:
    l_dset = len(dataset)
    samples_num = l_dset if l_dset < max_samples else max_samples
    indices = np.random.randint(0, l_dset, size=samples_num)
    samples = dataset[indices]

    tokenized_samples: list[Encoding] = tokenizer.encode_batch(
        samples[column_with_text]
    )
    samples_len = [len(sample.ids) for sample in tokenized_samples]

    return int(np.mean(samples_len))


class PreTrainGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset: Dataset,
        column_with_text: str,
        tokenizer: Tokenizer,
        seq_len: int,
        padding_token: str = "<pad>",
        mask_token: str = "<mask>",
        batch_size: int = 32,
        shuffle_seed: Union[None, int] = None,
        lora_option: int = -1,
        masked_percentage: float = 0.30,
        determine_num: int = 10**3,
    ):
        self.DETERMINE_NUM = determine_num
        self.MASKED_RATIO = 0.8
        self.REPLACED_RATIO = 1 - self.MASKED_RATIO

        self.dset = dataset
        self.column_with_text = column_with_text
        self.seq_len = seq_len
        self.masked_percentage = masked_percentage
        self.lora_option = lora_option

        if shuffle_seed is None:
            self.rnd_generator = None
        else:
            self.rnd_generator = np.random.default_rng(shuffle_seed)

        self.vocab_size = tokenizer.get_vocab_size()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.tokenizer.pad_token = padding_token
        self.tokenizer.mask_token = mask_token
        self.mask_id = self.tokenizer.mask_token_id

        avg_seq_len = _determine_avg_seq_len(
            dataset, column_with_text, tokenizer, self.DETERMINE_NUM
        )
        batch_factor = seq_len / avg_seq_len
        self.batch_size = math.ceil(batch_size * batch_factor)
        self.batch_size = min(self.batch_size, batch_size)

        self.__len = math.ceil(len(dataset) / self.batch_size)

        self.on_epoch_end()

    def __shuffle(self) -> None:
        dset = self.dset.shuffle(generator=self.rnd_generator)
        dset = dset.flatten_indices()
        self.dset = dset

    def __mask_replace_tokens(
        self, tokens_seqs: np.ndarray, masks: np.ndarray, vocab_size: int
    ) -> np.ndarray:
        masks = masks.astype(np.bool_)
        for i in range(len(tokens_seqs)):
            # data
            mask = masks[i]
            tokens = tokens_seqs[i]
            tokens, padding = tokens[mask], tokens[~mask]

            # ratios
            length = len(tokens)
            mask_prop = self.masked_percentage * self.MASKED_RATIO
            replace_prop = self.masked_percentage * self.REPLACED_RATIO

            tokens_to_mask = np.random.choice(
                a=[False, True],
                size=length,
                p=[1 - mask_prop, mask_prop],
            )
            tokens_to_replace = np.random.choice(
                a=[False, True],
                size=length,
                p=[1 - replace_prop, replace_prop],
            )
            cross = tokens_to_mask & tokens_to_replace
            tokens_to_replace = tokens_to_replace & ~cross

            # change
            replace = np.random.randint(0, vocab_size, size=length)
            masked = np.full(length, fill_value=self.mask_id, dtype=np.int32)

            tokens[tokens_to_replace] = replace[tokens_to_replace]
            tokens[tokens_to_mask] = masked[tokens_to_mask]
            tokens_seqs[i] = np.concatenate([tokens, padding])

        return tokens_seqs

    def __prepare_to_train(self, batch: list[str]) -> tuple[list[tf.Tensor], tf.Tensor]:
        # tokenize
        tokenized_batch: BatchEncoding = self.tokenizer(
            batch,
            return_tensors="np",
            padding=True,
        )
        sequences: np.ndarray = tokenized_batch["input_ids"]  # type: ignore
        masks: np.ndarray = tokenized_batch["attention_mask"]  # type: ignore

        # adjust_size
        pad_token_id = self.tokenizer.pad_token_id
        shape = list(sequences.shape)
        if shape[1] % self.seq_len != 0:
            shape[1] = self.seq_len - (shape[1] % self.seq_len)
            paddings = np.full(shape, fill_value=pad_token_id, dtype=np.int32)
            zeros = np.zeros(shape, dtype=np.int8)
            sequences = np.concatenate([sequences, paddings], axis=1)
            masks = np.concatenate([masks, zeros], axis=1)

        # reshape
        shape = sequences.shape
        scale = shape[1] // self.seq_len
        shape = [shape[0] * scale, self.seq_len]

        sequences = sequences.reshape(shape)
        masks = masks.reshape(shape)

        # remove blank
        rows_with_data = masks.any(axis=1)
        sequences = sequences[rows_with_data]
        masks = masks[rows_with_data]

        # lora
        lora = np.full((masks.shape[0], 1), fill_value=self.lora_option, dtype=np.int8)

        # data
        lables = tf.one_hot(sequences, self.vocab_size)

        seq = self.__mask_replace_tokens(sequences, masks, self.vocab_size)
        seq = tf.convert_to_tensor(seq, dtype=tf.int32)
        mask = tf.convert_to_tensor(masks, dtype=tf.int8)
        lora = tf.convert_to_tensor(lora, dtype=tf.int32)

        return [seq, mask, lora], lables

    def on_epoch_end(self) -> None:
        if self.rnd_generator is not None:
            self.__shuffle()

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        s = index * self.batch_size
        e = s + self.batch_size
        table = self.dset[s:e]
        batch: list[str] = table[self.column_with_text]
        return self.__prepare_to_train(batch)
