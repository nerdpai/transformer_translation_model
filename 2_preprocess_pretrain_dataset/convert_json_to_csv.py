import json
import csv
import mmap
import shutil
import pandas as pd
from pathlib import Path


def calculate_rows_for_each_csv(json_file: Path, size_of_each_csv_in_mb: int) -> int:
    MB_SIZE = 1024 * 1024
    with open(json_file, "r+") as f:
        mm = mmap.mmap(f.fileno(), 0)
        num_objects = sum(1 for line in iter(mm.readline, b""))
    file_stats = json_file.stat()
    size_in_mb = file_stats.st_size / (MB_SIZE)
    return int(num_objects * size_of_each_csv_in_mb / size_in_mb)


def json_to_csv(
    json_files: list[Path], csv_directory: Path, size_of_each_csv_in_mb: int = 30
) -> None:
    last_file_i = 0
    for file in json_files:
        rows_num = calculate_rows_for_each_csv(file, size_of_each_csv_in_mb)

        with open(file, "r") as f:
            write_header = False
            csv_file = Path("")
            for i, line in enumerate(f):
                if i % rows_num == 0:
                    csv_file = csv_directory / (str(last_file_i) + ".csv")
                    write_header = True
                    last_file_i += 1
                data = json.loads(line)

                reduced_data = {
                    key: data[key] for key in ("language", "raw_content") if key in data
                }

                reduced_data["raw_content"] = reduced_data["raw_content"].replace(
                    "\n", "\\n"
                )

                temp_df = pd.json_normalize(reduced_data)
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


def execute(json_files: list[Path], csv_directory: Path, each_csv_mb: int) -> None:
    is_place_good = __is_prepared(csv_directory)
    if is_place_good:
        return
    else:
        if csv_directory.exists():
            shutil.rmtree(csv_directory)

    csv_directory.mkdir(parents=True, exist_ok=True)

    json_to_csv(json_files, csv_directory, each_csv_mb)
