from pathlib import Path
from typing import TypeAlias

ModelMetrics: TypeAlias = tuple[str, dict[str, float]]


def save_metrics(save_dir: Path, metrics: list[ModelMetrics], encoding: str) -> None:
    with open(save_dir / "metrics.txt", "w", encoding=encoding) as f:
        for model, metric in metrics:
            f.write(f"{model}: {metric}\n")
