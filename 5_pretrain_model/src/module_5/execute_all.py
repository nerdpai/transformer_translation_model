import datasets
import shutil
from tensorflow._api.v2.v2 import keras
from tokenizers import Tokenizer

from module_5.specs_management.model_specs import ModelSpecs

import module_5.preparations.train_comps as prep_train_comps
import module_5.preparations.check_place as prep_check_place
import module_5.preparations.dataset as prep_dataset
import module_5.preparations.embedding as prep_embedding
import module_5.preparations.set_random as prep_random
import module_5.preparations.tokenizer as prep_tokenizer
import module_5.preparations.generator as prep_generator
import module_5.model.mbart_like_model as prep_model
import module_5.changeable as ch

import module_5.train_model as train_model
import module_5.save_model as save_model
import module_5.test_model as test_model
import module_5.models_metrics as models_metrics


def execute() -> None:
    prep_random.execute(ch.SEED)

    tokenizer = prep_tokenizer.execute(ch.tokenizer_path)
    dset = prep_dataset.execute(
        ch.datasets,
        ch.cc_mined_dir,
        ch.dset_cache_dir,
        ch.CONTENT_COLUMN,
        ch.DSET_BATCH_SIZE,
        ch.LANGS,
        ch.C4_SIZES,
    )

    test_results: list[models_metrics.ModelMetrics] = []

    for model_specs in ch.models_specs:
        emb = prep_embedding.execute(ch.emb_path)
        train_gen, test_gen = get_generators(dset, tokenizer, model_specs)
        train_comps = prep_train_comps.execute(
            ch.INIT_LR,
            ch.FINAL_LR,
            len(train_gen) * ch.EPOCHS,
            ch.PATIENCE,
            ch.PATIENCE_MONITOR,
        )
        model = get_model(model_specs, emb)

        save_dir = ch.save_dir / model_specs.name
        model = train_model.execute(
            model,
            train_gen,
            train_comps,
            ch.EPOCHS,
            save_dir,
        )

        if ch.SAVE_MODEL:
            save_model.execute(model, save_dir)

        metrics = test_model.execute(model, test_gen)
        test_results.append(models_metrics.ModelMetrics((model_specs.name, metrics)))

        del model
        keras.backend.clear_session()

    models_metrics.save_metrics(ch.save_dir, test_results, ch.TEXT_ENCODING)


def get_generators(
    dset: datasets.Dataset, tokenizer: Tokenizer, model_specs: ModelSpecs
) -> tuple[prep_generator.PreTrainGenerator, prep_generator.PreTrainGenerator]:
    dset_dict = dset.train_test_split(test_size=ch.TEST_SPLIT)
    test_dset: datasets.Dataset = dset_dict["test"]
    train_dset: datasets.Dataset = dset_dict["train"]

    train_gen, test_gen = [
        prep_generator.PreTrainGenerator(
            dataset=dset,
            column_with_text=ch.CONTENT_COLUMN,
            tokenizer=tokenizer,
            seq_len=model_specs.seq_len,
            padding_token=ch.PADDING_TOKEN,
            mask_token=ch.MASK_TOKEN,
            batch_size=ch.BATCH_SIZE,
            shuffle_seed=seed,
        )
        for dset, seed in zip([train_dset, test_dset], [ch.SEED, None])
    ]

    return train_gen, test_gen


def get_model(
    model_specs: ModelSpecs, emb_layer: keras.layers.Embedding
) -> keras.Model:
    emb_dim: int = emb_layer.output_dim

    coder_specs = prep_model.CoderSpecs(
        coder_layers_num=model_specs.transformer_layers_num,
        heads_num=model_specs.heads_num,
        is_causal=model_specs.causal,
        emb_dim=emb_dim,
        seq_len=model_specs.seq_len,
    )
    lora_specs = prep_model.LoraSpecs(
        loras_num=model_specs.lora_num,
        rank=model_specs.lora_rank,
        alpha=model_specs.lora_alpha,
    )
    layer_specs = prep_model.LayerSpecs(
        trainable=True,
        name="mbart_like",
        dtype=model_specs.dtype,
    )

    return prep_model.get_mbart_like_model(
        prep_model.MBartSpecs(
            emb_layer=emb_layer,
            coder_specs=coder_specs,
            lora_specs=lora_specs,
            layer_specs=layer_specs,
        )
    )


if __name__ == "__main__":
    if prep_check_place.execute(ch.save_dir):
        shutil.rmtree(ch.save_dir)
        ch.save_dir.mkdir(parents=True, exist_ok=True)

        execute()
