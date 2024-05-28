import matplotlib.pyplot as plt
from pathlib import Path

from specs_management.compare_specs import CompareSpecs


def execute(comp_specs: list[CompareSpecs], analitics_dir: Path) -> None:
    if len(comp_specs) == 0:
        return

    model_names = [spec.name for spec in comp_specs]
    analitics_dir.mkdir(parents=True, exist_ok=True)

    metrics_num = len(comp_specs[0].metrics)
    for i in range(metrics_num):
        values = [spec.metrics[i] for spec in comp_specs]
        name_of_value = comp_specs[0].metrics_names[i]

        plt.bar(model_names, values)
        plt.xlabel("Models")
        plt.ylabel(name_of_value)
        plt.title(f"{name_of_value} comparison")
        plt.savefig(analitics_dir / f"{name_of_value}_comparison.png")
        plt.close()
