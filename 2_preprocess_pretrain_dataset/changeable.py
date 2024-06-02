from pathlib import Path


ENCODING: str = "utf-8"


## gz
gz_file_pathes: list[Path] = [
    # elsewhere
]


## json
json_file_pathes: list[Path] = [
    # elsewhere
]
csv_directory: Path = Path(
    # elsewhere
)

TARGET_SIZE_OF_CSV_IN_MB: int = 30
CONTENT_KEY: str = "raw_content"
KEYS_TO_KEEP: list[str] = ["language", CONTENT_KEY]
