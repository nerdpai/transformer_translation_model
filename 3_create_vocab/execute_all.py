import prepare_place
import prepare_dataset
import prepare_tokenizer
import prepare_trainer
import train_tokenizer
import changeable as ch


if __name__ == "__main__":
    contin = prepare_place.execute(ch.save_tokenizer_dir)
    if contin:
        tokenizer = prepare_tokenizer.execute(
            ch.BOS_TOKEN,
            ch.EOS_TOKEN,
            ch.NEW_LINE_TOKEN,
            ch.UNK_TOKEN,
            ch.special_tokens,
        )
        dataset = prepare_dataset.execute(
            ch.cc_mined_dir, ch.cache_dir, ch.CONTENT_COLUMN, ch.BATCH_SIZE
        )
        trainer = prepare_trainer.execute(
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
