from tensorflow._api.v2.v2 import keras
from pathlib import Path


def execute(emb_path: Path) -> keras.layers.Embedding:
    model: keras.Sequential = keras.models.load_model(emb_path, compile=False)  # type: ignore
    return model.layers[0]
