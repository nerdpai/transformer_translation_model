import preparations.generator as prep_gen
import preparations.tokenizer as prep_tokenizer
import preparations.check_place as prep_check_place
import preparations.dataset as prep_dataset
import preparations.callbacks as prep_callbacks
import preparations.set_random as prep_random
import preparations.tf_config as prep_tf_config
import train_embedding
import changeable as ch


def train():
    prep_tf_config.execute()

    prep_random.execute(ch.SEED)

    tokenizer = prep_tokenizer.execute(tokenizer_path=ch.tokenizer_path)

    dataset = prep_dataset.execute(
        batch_size=ch.DATASET_BATCH_SIZE,
        c4_size=ch.c4_sizes,
        cache_dir=ch.dataset_cache_dir,
        content_column=ch.CONTENT_COLUMN,
        langs=ch.langs,
        cc_mined_dir=ch.cc_mined_dir,
        shuffle=ch.SEED,
    )

    generator_specs = ch.get_skip_gen_specs(dataset, tokenizer)
    generator = prep_gen.execute(generator_specs)

    callbacks = prep_callbacks.execute(
        ch.INIT_LR,
        ch.FINAL_LR,
        ch.EPOCS_NUM,
        ch.PARTS_PER_EPOCH,
        ch.PART_SIZE,
        ch.SAMPLES_PER_LINE_,
        ch.TRAIN_BATCH_SIZE,
        ch.EPOCH_PATIENT,
    )

    train_embedding.execute(
        generator=generator,
        tokenizer=tokenizer,
        embedding_dim=ch.EMBED_DIM,
        epochs_num=ch.EPOCS_NUM,
        output_path=ch.embedding_dir,
        pad_token=ch.PAD_TOKEN,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    if prep_check_place.execute(ch.embedding_dir):
        train()
