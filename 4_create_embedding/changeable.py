from typing import Callable
from pathlib import Path
from datasets import Dataset
from tokenizers import Tokenizer

from generator.skip_grams_gen import SkipGenSpecs


SEED: int = 42

## hidden
__DATASET_NUM_OF_LINES: int = 5 * 10**7
__EST_PART_SIZE: int = 10**3
__PARTS_NUM: int = __DATASET_NUM_OF_LINES // __EST_PART_SIZE
SAMPLES_PER_LINE_: int = 1500


## tokenizer
tokenizer_path = Path(
    # elsewhere
)


## dataset
cc_mined_dir = Path(
    # elsewhere
)
dataset_cache_dir = Path(
    # elsewhere
)

DATASET_BATCH_SIZE: int = 10**4
CONTENT_COLUMN: str = "raw_content"

c4_sizes: list[int] = [13 * 10**6, 7 * 10**6, 7 * 10**6]
langs: list[str] = ["en", "de", "fr"]


## generator_specs
generator_cache_dir = Path(str(dataset_cache_dir / "generator"))

TRAIN_BATCH_SIZE: int = 2**16
WINDOW_SIZE: int = 2
NUM_NEGATIVE_SAMPLES: int = 5
PART_SIZE: int = __DATASET_NUM_OF_LINES // __PARTS_NUM
PARTS_PER_EPOCH: int = int(2 * 10**-3 * __PARTS_NUM)
PART_INTERFERE: int = int(0.3 * PARTS_PER_EPOCH)
T: float = 10e-4
SHUFFLE: bool = True

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
embedding_dir = Path(
    # elsewhere
)
__TRAINING_LOOPS_NUM: int = 2

EPOCS_NUM: int = int(__TRAINING_LOOPS_NUM * __PARTS_NUM / PARTS_PER_EPOCH)

EMBED_DIM: int = 512
PAD_TOKEN: str = "<pad>"


## callbacks
INIT_LR: float = 10e-3
FINAL_LR: float = 10e-5
EPOCH_PATIENT: int = max(10, int(0.1 * EPOCS_NUM / __TRAINING_LOOPS_NUM))
