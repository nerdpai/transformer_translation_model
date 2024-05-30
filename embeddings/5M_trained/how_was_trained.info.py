from typing import Callable
from pathlib import Path
from datasets import Dataset
from tokenizers import Tokenizer

from generator.skip_grams_gen import SkipGenSpecs


SEED: int = 42

## hidden
__DATASET_NUM_OF_LINES: int = 5 * 10**6
__EST_PART_SIZE = 5 * 10**3
__PARTS_NUM: int = __DATASET_NUM_OF_LINES // __EST_PART_SIZE
SAMPLES_PER_LINE_: int = 1500


## tokenizer
tokenizer_path = Path("./3_create_vocab/vocabs/small/all/tokenizer.json")


## dataset
cc_mined_dir = Path("")
dataset_cache_dir = Path("./cache")

DATASET_BATCH_SIZE: int = 10**4
CONTENT_COLUMN: str = "raw_content"

c4_sizes: list[int] = [2 * 10**6, int(1.5 * 10**6), int(1.5 * 10**6)]
langs: list[str] = ["en", "de", "fr"]


## generator_specs
generator_cache_dir = Path(str(dataset_cache_dir / "generator"))

TRAIN_BATCH_SIZE: int = 2**16
WINDOW_SIZE: int = 2
NUM_NEGATIVE_SAMPLES: int = 5
PART_SIZE: int = __DATASET_NUM_OF_LINES // __PARTS_NUM
PARTS_PER_EPOCH: int = int(2 * 10**-2 * __PARTS_NUM)
PART_INTERFERE: int = int(0.34 * PARTS_PER_EPOCH)
T: float = 10e-4
SHUFFLE: bool = False

get_skip_gen_specs: Callable[[Dataset, Tokenizer], SkipGenSpecs] = (
    lambda dataset, tokenizer: SkipGenSpecs(
        dataset=dataset,
        column_with_text=CONTENT_COLUMN,
        tokenizer=tokenizer,
        cache_dir=generator_cache_dir,
        batch_size=TRAIN_BATCH_SIZE,
        window_size=WINDOW_SIZE,
        num_ns=NUM_NEGATIVE_SAMPLES,
        part_size=PART_SIZE,
        parts_per_epoch=PARTS_PER_EPOCH,
        part_interfere=PART_INTERFERE,
        t=T,
        shuffle=SHUFFLE,
    )
)


## train_embedding
embedding_dir = Path("./small_all_shuffled_emb")
__TRAINING_LOOPS_NUM: int = 1

EPOCS_NUM: int = int(__TRAINING_LOOPS_NUM * __PARTS_NUM / PARTS_PER_EPOCH)

EMBED_DIM: int = 512
PAD_TOKEN: str = "<pad>"


## callbacks
INIT_LR: float = 10e-3
FINAL_LR: float = 10e-4
EPOCH_PATIENT: int = EPOCS_NUM