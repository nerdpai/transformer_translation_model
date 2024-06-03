from pathlib import Path


def execute(gz_dir: Path, json_dir: Path) -> tuple[list[Path], list[Path]]:
    gz_files = list(gz_dir.rglob("*.gz"))
    json_files: list[Path] = []

    for gz_file in gz_files:

        json_file = json_dir / gz_file.relative_to(gz_dir).with_suffix(".json")
        json_files.append(json_file)

    return (gz_files, json_files)
