from tensorflow._api.v2.v2 import keras


from module_4_5.preparations.dataset_components.generator import NeighbourGenerator


def execute(
    model: keras.Model, test_gen: NeighbourGenerator
) -> tuple[list[float], list[str]]:
    metrics = model.evaluate(test_gen)  # type: ignore
    if not isinstance(metrics, list):
        metrics = [metrics]
    metrics_names = model.metrics_names
    return metrics, metrics_names  # type: ignore
