from pathlib import Path

## tokenizer
NEW_LINE_TOKEN: str = "<newline>"
UNK_TOKEN: str = "<unk>"
BOS_TOKEN: str = "<s>"
EOS_TOKEN: str = "</s>"
special_tokens: list[str] = [
    BOS_TOKEN,
    "<pad>",
    EOS_TOKEN,
    "<mask>",
    NEW_LINE_TOKEN,
    UNK_TOKEN,
]

##dataset
cc_mined_dir = Path(
    # elsewhere
)
cache_dir = Path(
    # elsewhere
)
BATCH_SIZE: int = 10000
CONTENT_COLUMN: str = "raw_content"
c4_sizes: list[int] = [10**7, 3 * 10**6, 3 * 10**6]
langs: list[str] = ["en", "de", "fr"]

## trainer
MAX_TOKEN_LENGTH: int = 2**4
MIN_FREQUENCY: int = 2
VOCAB_SIZE: int = 2**16

## train_tokenizer
save_tokenizer_dir = Path(
    # elsewhere
)
