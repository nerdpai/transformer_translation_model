from pathlib import Path
from tensorflow._api.v2.v2 import keras


def execute(model: keras.Model, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "model.h5"))
