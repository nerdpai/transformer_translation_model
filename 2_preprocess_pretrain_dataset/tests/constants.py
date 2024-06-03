from pathlib import Path


ENCODING: str = "utf-8"
__DATA_PATH = Path(__file__).parent / "data"


## dirs
test_dir: Path = __DATA_PATH / "test_data"
true_dir: Path = __DATA_PATH / "true_data"

gz_dir_name = "gz"
json_dir_name = "json"
csv_dir_name = "csv"

TARGET_SIZE_OF_CSV_IN_MB: int = 30
CONTENT_KEY: str = "raw_content"
KEYS_TO_KEEP: list[str] = ["language", CONTENT_KEY]
