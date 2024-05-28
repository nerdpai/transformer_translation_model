class CompareSpecs:
    def __init__(
        self,
        name: str,
        metrics: list[float],
        metrics_names: list[str],
    ):
        self.name = name
        self.metrics = metrics
        self.metrics_names = metrics_names
