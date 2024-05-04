from tokenizers import trainers, pre_tokenizers


def execute(
    vocab_size: int, special_tokens: list[str], min_freq: int, max_token_l: int
) -> trainers.BpeTrainer:
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_freq,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        max_token_length=max_token_l,
    )  # type: ignore
    return trainer
