import module_3.preparations.dataset as prep_dataset
import module_3.preparations.check_place as prep_check_place
import module_3.preparations.tokenizer as prep_tokenizer
import module_3.preparations.trainer as prep_trainer
import module_3.train_tokenizer as train_tokenizer
import module_3.changeable as ch


def train() -> None:
    tokenizer = prep_tokenizer.execute(
        ch.BOS_TOKEN,
        ch.EOS_TOKEN,
        ch.NEW_LINE_TOKEN,
        ch.UNK_TOKEN,
        ch.special_tokens,
    )
    dataset = prep_dataset.execute(
        ch.cc_mined_dir,
        ch.cache_dir,
        ch.CONTENT_COLUMN,
        ch.BATCH_SIZE,
        ch.langs,
        ch.c4_sizes,
    )
    trainer = prep_trainer.execute(
        ch.VOCAB_SIZE, ch.special_tokens, ch.MIN_FREQUENCY, ch.MAX_TOKEN_LENGTH
    )
    train_tokenizer.execute(
        tokenizer,
        trainer,
        dataset,
        ch.BATCH_SIZE,
        ch.save_tokenizer_dir,
        ch.CONTENT_COLUMN,
    )


if __name__ == "__main__":
    contin = prep_check_place.execute(ch.save_tokenizer_dir)
    if contin:
        train()
