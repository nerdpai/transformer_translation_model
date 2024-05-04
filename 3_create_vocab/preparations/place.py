from pathlib import Path


def prepare_place(save_dir: Path) -> bool:
    if save_dir.exists():
        overwrite = input(
            "Tokenizer already exists in the directory. Do you want to overwrite it? (y/n): "
        )
        if overwrite.lower() != "y":
            print("Early stopping.")
            return False

    save_dir.mkdir(parents=True, exist_ok=True)
    return True


def execute(save_dir: Path) -> bool:
    return prepare_place(save_dir)
