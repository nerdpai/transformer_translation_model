from tensorflow._api.v2.v2 import keras

from module_5.preparations.generator import PreTrainGenerator


def execute(
    model: keras.Model,
    generator: PreTrainGenerator,
) -> dict[str, float]:

    data: dict[str, float] = model.evaluate(generator, return_dict=True)  # type: ignore

    return data
