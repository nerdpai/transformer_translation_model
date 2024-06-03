import json
import csv
import mmap
import shutil
import pandas as pd
from pathlib import Path


def calculate_rows_for_each_csv(
    json_file: Path,
    size_of_each_csv_in_mb: int,
    encoding: str,
) -> int:
    MB_SIZE: int = 1024 * 1024
    with open(json_file, "r+", encoding=encoding) as f:
        mm = mmap.mmap(f.fileno(), 0)
        num_objects = sum(1 for line in iter(mm.readline, b""))
    file_stats = json_file.stat()
    size_in_mb = file_stats.st_size / (MB_SIZE)
    return int(num_objects * size_of_each_csv_in_mb / size_in_mb)


def json_to_csv(
    json_file: Path,
    csv_directory: Path,
    keys_to_keep: list[str],
    content_key: str,
    encoding: str,
    size_of_each_csv_in_mb: int = 30,
) -> None:
    last_file_i = 0
    rows_num = calculate_rows_for_each_csv(json_file, size_of_each_csv_in_mb, encoding)

    with open(json_file, "r", encoding=encoding) as f:
        write_header = False
        csv_file = None
        for i, line in enumerate(f):
            if i % rows_num == 0:
                csv_file = csv_directory / (str(last_file_i) + ".csv")
                write_header = True
                last_file_i += 1
            data = json.loads(line)

            reduced_data = {key: data[key] for key in keys_to_keep if key in data}

            reduced_data[content_key] = reduced_data[content_key].replace("\n", "\\n")

            temp_df = pd.json_normalize(reduced_data)

            if csv_file is None:
                raise ValueError("path for csv is None")

            temp_df.to_csv(
                csv_file,
                mode="a",
                header=write_header,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar="\\",
            )

            write_header = False


def __is_prepared(save_dir: Path) -> bool:
    if save_dir.exists():
        user_input = input(
            "The directory already exists. Do you want to delete it and its contents? (y/n) [n]: "
        )
        if user_input.lower() != "y":
            print("Early stopping.")
            return True

    return False


def prepare_place(csv_directory: Path, delete_anyway: bool) -> None:
    if not delete_anyway:
        is_place_good = __is_prepared(csv_directory)
        if is_place_good:
            exit()

    if csv_directory.exists():
        shutil.rmtree(csv_directory)
    csv_directory.mkdir(parents=True, exist_ok=True)


def execute(
    json_file: Path,
    csv_directory: Path,
    keys_to_keep: list[str],
    content_key: str,
    encoding: str,
    each_csv_mb: int,
    delete_anyway: bool = False,
) -> None:
    prepare_place(csv_directory, delete_anyway)
    json_to_csv(
        json_file,
        csv_directory,
        keys_to_keep,
        content_key,
        encoding,
        each_csv_mb,
    )
