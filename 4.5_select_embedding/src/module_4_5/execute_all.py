from tensorflow._api.v2.v2 import keras

import module_4_5.preparations.train_comps as train_comps
import module_4_5.preparations.check_place as prep_check_place
import module_4_5.preparations.dataset as prep_dataset
import module_4_5.preparations.embedding as prep_embedding
import module_4_5.preparations.set_random as prep_random
import module_4_5.preparations.tokenizer as prep_tokenizer
import module_4_5.changeable as ch

from module_4_5.specs_management.emb_train_specs import EmbTrainSpecs
from module_4_5.specs_management.compare_specs import CompareSpecs
from module_4_5.preparations.dataset_components.generator import NeighbourGenerator

import module_4_5.train_predict_model as train_model
import module_4_5.test_model as test_model
import module_4_5.compare_models as compare_models


def execute() -> None:
    prep_random.execute(ch.SEED)

    compare_specs: list[CompareSpecs] = []

    for train_spec in ch.TRAIN_SPECS:
        print(f"Preparing generator for {train_spec.name}")
        train_gen, test_gen = get_generators(train_spec)
        print(f"Training {train_spec.name}")
        model = train(train_spec, train_gen)
        print(f"Testing {train_spec.name}")
        metrics, metrics_names = test_model.execute(model, test_gen)

        comp_specs = CompareSpecs(train_spec.name, metrics, metrics_names)
        compare_specs.append(comp_specs)

    compare_models.execute(compare_specs, ch.analitics_dir)


def get_generators(
    train_spec: EmbTrainSpecs,
) -> tuple[NeighbourGenerator, NeighbourGenerator]:
    tokenizer = prep_tokenizer.execute(train_spec.tokenizer_path)

    fetch_specs = prep_dataset.FetcherSpecs(
        cache_dir=ch.cache_dir / "dataset",
        content_column=ch.CONTENT_COLUMN,
        langs=train_spec.langs,
        subset_of_train=train_spec.subset_size,
        subset_of_test=1.0,
        seed=ch.SEED,
    )
    gen_specs = prep_dataset.GeneratorSpecs(
        cache_dir=ch.cache_dir / train_spec.name,
        tokenizer=tokenizer,
        transform_batch_size=ch.DATASET_BATCH_SIZE,
        train_batch_size=ch.TRAIN_BATCH_SIZE,
        window_size=ch.WINDOW_SIZE,
        shuffle=True,
        shuffle_overlap=ch.SHUFFLE_OVERLAP,
    )
    train_gen, test_gen = prep_dataset.execute(fetch_specs, gen_specs)

    return train_gen, test_gen


def train(train_spec: EmbTrainSpecs, train_gen: NeighbourGenerator) -> keras.Model:

    emb = prep_embedding.execute(train_spec.emb_path)

    comp_elements = train_comps.execute(
        init_lr=ch.INITIAL_LR,
        final_lr=ch.FINAL_LR,
        patience_in_epochs=ch.PATIENCE_IN_EPOCHS,
        patience_monitor=ch.PATIENCE_MONITOR,
    )

    history_dir = (ch.analitics_dir / "history") / train_spec.name

    model = train_model.execute(
        emb=emb,  # type: ignore
        generator=train_gen,
        comp_elements=comp_elements,
        epochs_num=ch.EPOCHS_NUM,
        history_dir=history_dir,
    )
    return model


if __name__ == "__main__":
    if prep_check_place.execute(ch.analitics_dir):
        execute()
