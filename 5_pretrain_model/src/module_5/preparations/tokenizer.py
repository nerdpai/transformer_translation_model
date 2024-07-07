from tokenizers import Tokenizer
from pathlib import Path


def execute(
    tokenizer_path: Path,
) -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
