import shutil
import datasets
import pandas as pd
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from pathlib import Path
from tqdm import tqdm


def __nan_remove(texts: list[str]) -> list[str]:
    df = pd.DataFrame(texts, columns=["data"])
    df = df[df["data"].notna()]
    return df["data"].tolist()


def tokenize_dataset(
    dataset: datasets.Dataset,
    tokenizer: Tokenizer,
    cache_dir: Path,
    column_with_sequence: str,
    batch_size: int = 0,
    tokenized_column_name: str = "",
) -> None:

    if cache_dir.exists():
        create_dir = input(
            "Dataset already tokenized. Do you want to rewrite it? (y/n): "
        )
        if create_dir.lower() != "y":
            print("Stopping tokenization.")
            return
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    if batch_size <= 0:
        batch_size = len(dataset)
    if tokenized_column_name == "":
        tokenized_column_name = column_with_sequence

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    for i in tqdm(range(0, len(dataset), batch_size), desc="Tokenization"):
        data = dataset[i : i + batch_size]
        texts: list[str] = data[column_with_sequence]

        texts = __nan_remove(texts)
        if len(texts) == 0:
            continue

        tokenized = fast_tokenizer(texts, add_special_tokens=True)["input_ids"]  # type: ignore

        dset: datasets.Dataset = datasets.Dataset.from_dict(
            {tokenized_column_name: tokenized}
        )
        dset.save_to_disk(str(cache_dir / f"{i}.data"))
