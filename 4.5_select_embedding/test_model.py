from tensorflow.keras import Model
from typing import Tuple

from preparations.dataset_components.generator import NeighbourGenerator


def execute(
    model: Model, test_gen: NeighbourGenerator
) -> Tuple[list[float], list[str]]:
    metrics = model.evaluate(test_gen)  # type: ignore
    if not isinstance(metrics, list):
        metrics = [metrics]
    metrics_names = model.metrics_names
    return metrics, metrics_names  # type: ignore
