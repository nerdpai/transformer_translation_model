from pathlib import Path


def execute(json_dir: Path, csv_dir: Path) -> tuple[list[Path], list[Path]]:
    json_files = list(json_dir.rglob("*.json"))
    csv_each_json_dir: list[Path] = []

    for json_file in json_files:

        csv_for_json = csv_dir / json_file.relative_to(json_dir).with_suffix("")
        csv_each_json_dir.append(csv_for_json)

    return (json_files, csv_each_json_dir)
