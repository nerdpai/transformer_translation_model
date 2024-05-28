from pathlib import Path

from specs_management.create_specs import prepare_specs


SEED: int = 42


## hidden
__embs_dir = Path(
    # elsewhere
)
__EMB_CATEGORIES: list[str] = [
    # categories
]

__EMB_FILE: str = "emb.h5"

__tokenizers_dir = Path(
    # elsewhere
)
__TOKENIZER_FILE: str = "tokenizer.json"

__LANGS: list[str] = ["en", "de", "fr"]
__UNIVERSAL_LANG: str = "all"


## train_specs
TRAIN_SPECS = prepare_specs(
    __embs_dir,
    __EMB_FILE,
    __EMB_CATEGORIES,
    __LANGS,
    __UNIVERSAL_LANG,
    __tokenizers_dir,
    __TOKENIZER_FILE,
)


## dataset
cache_dir = Path(
    # elsewhere
)

CONTENT_COLUMN: str = "content"
DATASET_BATCH_SIZE: int = 10**4
SHUFFLE_OVERLAP: float = 0.34


## train
TRAIN_BATCH_SIZE: int = 10**3

WINDOW_SIZE: int = 2

INITIAL_LR: float = 10**-3
FINAL_LR: float = 10**-4

EPOCHS_NUM: int = 5
PATINET_IN_EPOCHS: int = 2

analitics_dir = Path(
    # elsewhere
)
