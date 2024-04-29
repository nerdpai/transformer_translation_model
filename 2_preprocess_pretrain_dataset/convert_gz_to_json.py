import gzip
import json
from pathlib import Path


def process_data(data: bytes, json_file_path: Path) -> None:
    json_str = data.decode("utf-8")

    json_data = json.loads(json_str)

    with json_file_path.open(mode="a") as json_file:
        json.dump(json_data, json_file)
        json_file.write("\n")


def gz_to_json(gz_file_path: Path) -> None:
    json_file_path = gz_file_path.with_suffix("")

    if json_file_path.is_file():
        user_input = input(
            "The JSON file already exists. Do you want to overwrite it? (y/n): "
        )
        if user_input.lower() == "y":
            json_file_path.unlink()
        else:
            print("Early stopping.")
            exit()

    with gzip.open(gz_file_path, "r") as compressed_file:
        data: bytes = b""

        for line in compressed_file:
            data += line

            if data.endswith(b"\n"):
                process_data(data, json_file_path)
                data = b""


def execute(gz_file_pathes: list[Path]) -> None:
    for gz_file_path in gz_file_pathes:
        gz_to_json(gz_file_path)
