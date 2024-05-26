from pathlib import Path


def check_place(save_dir: Path) -> bool:
    if save_dir.exists():
        user_input = input(
            "Tokenizer already exists in the directory. Do you want to overwrite it? (y/n) [n]: "
        )
        if user_input.lower() != "y":
            print("Early stopping.")
            return False

    return True


def execute(save_dir: Path) -> bool:
    return check_place(save_dir)
