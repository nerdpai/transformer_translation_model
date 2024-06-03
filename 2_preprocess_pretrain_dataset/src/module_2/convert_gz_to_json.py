import gzip
import json
from pathlib import Path


def __is_prepared(save_dir: Path) -> bool:
    if save_dir.exists():
        user_input = input(
            "The JSON file already exists. Do you want to overwrite it? (y/n) [n]: "
        )
        if user_input.lower() != "y":
            print("Early stopping.")
            return True

    return False


def prepare_place(json_file_path: Path, delete_anyway: bool) -> None:
    if not delete_anyway:
        is_place_good = __is_prepared(json_file_path)
        if is_place_good:
            exit()

    json_file_path.unlink(missing_ok=True)
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    json_file_path.touch()


def process_data(data: bytes, json_file_path: Path, encoding: str) -> None:
    json_str = data.decode(encoding)

    json_data = json.loads(json_str)

    with json_file_path.open(mode="a") as json_file:
        json.dump(json_data, json_file)
        json_file.write("\n")


def gz_to_json(gz_file_path: Path, json_file_path: Path, encoding: str) -> None:
    with gzip.open(gz_file_path, "r") as compressed_file:
        data: bytes = b""

        for line in compressed_file:
            data += line

            if data.endswith(b"\n"):
                process_data(data, json_file_path, encoding)
                data = b""


def execute(
    gz_file_path: Path,
    json_file_path: Path,
    encoding: str,
    delete_anyway: bool = False,
) -> None:
    prepare_place(json_file_path, delete_anyway)
    gz_to_json(gz_file_path, json_file_path, encoding)
