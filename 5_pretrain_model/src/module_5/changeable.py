import tensorflow._api.v2.v2 as tf
from pathlib import Path

from module_5.specs_management.model_specs import ModelSpecs
from module_5.specs_management.possible_datasets import DatasetsEnum


SEED = 42

# Dirs
save_dir = Path(
    # elsewhere
)
tokenizer_path = Path(
    # elsewhere
)
emb_path = Path(
    # elsewhere
)


# Models specs
SAVE_MODEL: bool = False
HEADS_NUM: int = 8
TRANSFORMER_NUM: int = 1
LORA_NUM: int = 1
LORA_RANK: int = 8
LORA_ALPHA: int = 32
CAUSAL: bool = True
MODEL_DTYPE: tf.DType = tf.float32
SEQ_LENS: list[int] = [i**2 for i in range(6, 10)]
NAMES: list[str] = [
    f"layers={TRANSFORMER_NUM}-heads={HEADS_NUM}-lora={LORA_NUM}-seq_len={i}-causal={CAUSAL}"
    for i in SEQ_LENS
]

models_specs: list[ModelSpecs] = [
    ModelSpecs(
        name=name,
        seq_len=seq_len,
        heads_num=HEADS_NUM,
        transformer_layers_num=TRANSFORMER_NUM,
        lora_num=LORA_NUM,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        causal=CAUSAL,
        dtype=MODEL_DTYPE,
    )
    for name, seq_len in zip(NAMES, SEQ_LENS)
]


# Datasets specs
datasets: list[DatasetsEnum] = [
    DatasetsEnum.C4,
]

cc_mined_dir = Path(
    # elsewhere
)

dset_cache_dir = Path(
    # elsewhere
)

CONTENT_COLUMN: str = "raw_content"
BATCH_SIZE: int = 4
LANGS: list[str] = ["en", "fr", "de"]
DSET_BATCH_SIZE: int = 10**3
C4_SIZES: list[int] = [10**6 * 3, 10**6, 10**6]
PADDING_TOKEN: str = "<pad>"
MASK_TOKEN: str = "<mask>"
TEST_SPLIT: float = 0.2


# Training specs
INIT_LR: float = 1e-2
FINAL_LR: float = 1e-5
EPOCHS: int = 2
PATIENCE: int = 100
PATIENCE_MONITOR: str = "loss"


# Save metrics
TEXT_ENCODING: str = "utf-8"
