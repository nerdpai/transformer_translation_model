import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

from module_5.model.mbart_like_model import (
    get_mbart_like_model,
    MBartSpecs,
    CoderSpecs,
    LoraSpecs,
    LayerSpecs,
)


class Data(NamedTuple):
    seq: tf.Tensor
    mask: tf.Tensor
    lora: tf.Tensor
    target: tf.Tensor


@dataclass(frozen=True)
class Constants:
    emb_dim: int = 32
    heads_num: int = 4
    is_causal: bool = True
    coder_layers_num: int = 3
    seq_len: int = 5
    rank: int = 8
    alpha: int = 32
    loras_num: int = 3
    trainable: bool = True
    name: str = "mbart"
    dtype: tf.dtypes.DType = tf.float32
    vocab_size: int = 12
    repeat: int = 3


def get_specs(consts: Constants) -> MBartSpecs:
    coder_specs = CoderSpecs(
        coder_layers_num=consts.coder_layers_num,
        heads_num=consts.heads_num,
        is_causal=consts.is_causal,
        emb_dim=consts.emb_dim,
        seq_len=consts.seq_len,
    )
    lora_specs = LoraSpecs(
        loras_num=consts.loras_num,
        alpha=consts.alpha,
        rank=consts.rank,
    )
    layer_specs = LayerSpecs(
        trainable=consts.trainable,
        name=consts.name,
        dtype=consts.dtype,
    )
    emb_layer = keras.layers.Embedding(
        input_dim=consts.vocab_size,
        output_dim=consts.emb_dim,
    )

    return MBartSpecs(
        emb_layer=emb_layer,
        coder_specs=coder_specs,
        lora_specs=lora_specs,
        layer_specs=layer_specs,
    )


def get_data(repeat: int) -> Data:
    seq = tf.convert_to_tensor([1, 2, 3, 0, 0], dtype=tf.int32)
    seq = tf.expand_dims(seq, axis=0)
    seq = tf.repeat(seq, repeat, axis=0)
    mask = tf.convert_to_tensor([1, 1, 1, 0, 0], dtype=tf.int8)
    mask = tf.expand_dims(mask, axis=0)
    mask = tf.repeat(mask, repeat, axis=0)
    lora = tf.convert_to_tensor([-1], dtype=tf.int32)
    lora = tf.expand_dims(lora, axis=0)
    lora = tf.repeat(lora, repeat, axis=0)
    target = tf.convert_to_tensor([5, 6, 7, 8, 0], dtype=tf.int32)
    target = tf.expand_dims(target, axis=0)
    target = tf.repeat(target, repeat, axis=0)
    target = tf.one_hot(target, depth=12)

    return Data(seq, mask, lora, target)


def get_tflite_path() -> Path:
    path = Path(__file__)
    path = path.parent

    user_input = input(f"Where to save the tflite model?[{path}]: ")
    if user_input != "":
        path = Path(user_input)

    return path / "model.tflite"


def save_tflite_model(model: keras.Model, file_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True

    tflite_model = converter.convert()
    with open(str(file_path), "wb") as f:
        f.write(tflite_model)


def test_interpreter(file_path: Path, data: Data) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(file_path))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    model_input = [data.seq, data.mask, data.lora]
    indixes_names = ["seq", "mask", "lora"]
    indixes: dict[str, int] = {}
    for i_details in input_details:
        for name in indixes_names:
            if name in i_details["name"]:
                indixes[name] = i_details["index"]
                break

    for name, val in zip(indixes_names, model_input):
        interpreter.set_tensor(indixes[name], val)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    print(output_data)


def main() -> None:
    consts = Constants()
    mbart_specs = get_specs(consts)
    model = get_mbart_like_model(mbart_specs)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalCrossentropy()],
    )

    data = get_data(consts.repeat)
    model.fit(x=[data.seq, data.mask, data.lora], y=data.target, epochs=2)

    tflite_path = get_tflite_path()
    save_tflite_model(model, tflite_path)

    interpreter_data = get_data(repeat=1)
    test_interpreter(tflite_path, interpreter_data)


if __name__ == "__main__":
    main()
