import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colormaps
from pathlib import Path

from specs_management.compare_specs import CompareSpecs


def execute(comp_specs: list[CompareSpecs], analitics_dir: Path) -> None:
    if len(comp_specs) == 0:
        return

    model_names = [spec.name for spec in comp_specs]
    analitics_dir.mkdir(parents=True, exist_ok=True)

    metrics_num = len(comp_specs[0].metrics)
    num_models = len(comp_specs)
    color_map = colormaps.get_cmap("tab20b")
    colors = color_map(np.linspace(0, 1, num_models))

    for i in range(metrics_num):
        values = [spec.metrics[i] for spec in comp_specs]
        name_of_value = comp_specs[0].metrics_names[i]

        ax: Axes
        _, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(range(1, len(comp_specs) + 1), values, color=colors)
        ax.set_xlabel("Models")
        ax.set_ylabel(name_of_value)
        ax.set_title(f"{name_of_value} comparison")
        ax.set_xticks(range(1, len(comp_specs) + 1))
        ax.legend(bars, model_names, bbox_to_anchor=(1.03, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(analitics_dir / f"{name_of_value}_comparison.png")
        plt.close()
