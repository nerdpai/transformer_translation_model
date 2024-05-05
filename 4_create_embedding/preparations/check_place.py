from pathlib import Path


def check_place(save_dir: Path) -> bool:
    if save_dir.exists():
        overwrite = input(
            "Embedding already exists in the directory. Do you want to overwrite it? (y/n): "
        )
        if overwrite.lower() != "y":
            print("Early stopping.")
            return False

    return True


def execute(save_dir: Path) -> bool:
    return check_place(save_dir)
