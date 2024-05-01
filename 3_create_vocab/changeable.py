from pathlib import Path

CONTENT_COLUMN = "raw_content"
cc_mined_dir = Path(
    # elsewhere
)
cache_dir = Path(
    # elsewhere
)
save_tokenizer_dir = Path(
    # elsewhere
)

NEW_LINE_TOKEN = "<newline>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

special_tokens = [BOS_TOKEN, "<pad>", EOS_TOKEN, "<mask>", NEW_LINE_TOKEN, UNK_TOKEN]
c4_sizes = [10**7, 3 * 10**6, 3 * 10**6]
langs = ["en", "de", "fr"]

MAX_TOKEN_LENGTH = 2**4
MIN_FREQUENCY = 2
VOCAB_SIZE = 2**16
BATCH_SIZE = 10000
