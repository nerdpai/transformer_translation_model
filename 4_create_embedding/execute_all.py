import preparations.generator as prep_gen
import preparations.tokenizer as prep_tokenizer
import preparations.check_place as prep_check_place
import preparations.dataset as prep_dataset
import train_embedding
import changeable


def train():
    tokenizer = prep_tokenizer.execute(tokenizer_path=changeable.tokenizer_path)

    dataset = prep_dataset.execute(
        batch_size=changeable.DATASET_BATCH_SIZE,
        c4_size=changeable.c4_sizes,
        cache_dir=changeable.dataset_cache_dir,
        content_column=changeable.CONTENT_COLUMN,
        langs=changeable.langs,
        cc_mined_dir=changeable.cc_mined_dir,
    )

    generator_specs = changeable.get_skip_gen_specs(dataset, tokenizer)

    generator = prep_gen.execute(generator_specs)

    train_embedding.execute(
        generator=generator,
        tokenizer=tokenizer,
        embedding_dim=changeable.EMBED_DIM,
        epochs_num=changeable.EPOCS_NUM,
        learning_rate=changeable.LEARNING_RATE,
        output_path=changeable.embedding_dir,
        pad_token=changeable.PAD_TOKEN,
    )


if __name__ == "__main__":
    if prep_check_place.execute(changeable.embedding_dir):
        train()
