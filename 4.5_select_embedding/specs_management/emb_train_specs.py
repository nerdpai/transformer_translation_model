from pathlib import Path


class EmbTrainSpecs:
    def __init__(
        self,
        name: str,
        emb_path: Path,
        tokenizer_path: Path,
        langs: list[str],
        subset_size: float = 1.0,
    ):
        self.name = name
        self.emb_path = emb_path
        self.tokenizer_path = tokenizer_path
        self.langs = langs
        self.subset_size = subset_size
