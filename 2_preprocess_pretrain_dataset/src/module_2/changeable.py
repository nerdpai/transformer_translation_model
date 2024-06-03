from pathlib import Path


ENCODING: str = "utf-8"


## dirs
gz_dir = Path("./2_preprocess_pretrain_dataset/tests/data/test_data/gz")
json_dir = Path("./2_preprocess_pretrain_dataset/tests/data/test_data/json")
csv_dir: Path = Path("./2_preprocess_pretrain_dataset/tests/data/test_data/csv")

TARGET_SIZE_OF_CSV_IN_MB: int = 30
CONTENT_KEY: str = "raw_content"
KEYS_TO_KEEP: list[str] = ["language", CONTENT_KEY]
