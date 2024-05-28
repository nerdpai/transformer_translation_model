from tensorflow.keras.layers import Embedding
from keras import Sequential
from pathlib import Path
from keras.models import load_model


def execute(emb_path: Path) -> Embedding:
    model: Sequential = load_model(emb_path)  # type: ignore
    return model.layers[0]
