#               |
# helpful link: v
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mbart/modeling_mbart.py


import tensorflow._api.v2.v2 as tf
from tensorflow._api.v2.v2 import keras
from dataclasses import dataclass, replace


from module_5.model.layers.transformer.encoder import Encoder, CoderSpecs
from module_5.model.layers.transformer.decoder import Decoder
from module_5.model.layers.lora import LoraSpecs
from module_5.model.layers.layer_specs import LayerSpecs
from module_5.model.layers.fully_connect import FullyConnected

# pylint: disable=e0401
from module_5.model.constants import WEIGHTS_INITIALIZER


@dataclass(frozen=True)
class MBartSpecs:
    emb_layer: keras.layers.Embedding
    coder_specs: CoderSpecs
    lora_specs: LoraSpecs
    layer_specs: LayerSpecs


class MBartLikeModel(keras.layers.Layer):
    def __init__(
        self,
        specs: MBartSpecs,
    ):
        self.specs = specs
        super(MBartLikeModel, self).__init__(
            specs.layer_specs.trainable,
            specs.layer_specs.name,
            specs.layer_specs.dtype,
            specs.layer_specs.dynamic,
        )

        self.emb_layer: keras.layers.Embedding = specs.emb_layer

        self.use_causal = specs.coder_specs.is_causal
        self.encoder_specs = replace(specs.coder_specs, is_causal=False)
        self.decoder_specs = replace(specs.coder_specs, is_causal=self.use_causal)

        self.encoder: Encoder
        self.decoder: Decoder
        self.classifier: FullyConnected

    def build(self, input_shape) -> None:
        super().build(input_shape)

        if not self.emb_layer.built:
            self.emb_layer.build(
                input_shape=tf.TensorShape((None, self.emb_layer.input_dim))
            )

        self.encoder = Encoder(
            self.encoder_specs,
            self.specs.lora_specs,
            replace(
                self.specs.layer_specs, name=f"{self.specs.layer_specs.name}_encoder"
            ),
        )
        self.encoder.build(input_shape)

        self.decoder = Decoder(
            self.decoder_specs,
            self.specs.lora_specs,
            replace(
                self.specs.layer_specs, name=f"{self.specs.layer_specs.name}_decoder"
            ),
        )
        self.decoder.build(input_shape)

        emb_weights = self.emb_layer.get_weights()
        in_dim, o_dim = emb_weights[0].shape[1], emb_weights[0].shape[0]
        classifier_shape = tf.TensorShape((in_dim, o_dim))
        self.classifier = FullyConnected(
            shape=classifier_shape,
            kernel_initializer=WEIGHTS_INITIALIZER,
            bias_initializer=None,
            lora_specs=self.specs.lora_specs,
            layer_specs=replace(
                self.specs.layer_specs, name=f"{self.specs.layer_specs.name}_classifier"
            ),
        )
        self.classifier.build(input_shape)

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], *args, **kwargs
    ) -> tf.Tensor:
        sequence, mask, condition = inputs  # bs*s, bs*s, bs*1
        sequence = self.emb_layer(sequence)  # bs*s*e
        encoded = self.encoder([sequence, mask, condition])  # bs*s*e
        decoded = self.decoder([encoded, sequence, mask, condition])  # bs*s*e

        logits = self.classifier([decoded, condition])  # bs*s*v

        return logits


def get_mbart_like_model(mbart_specs: MBartSpecs) -> keras.Model:
    seq_inp = keras.Input(
        shape=(mbart_specs.coder_specs.seq_len,),
        dtype=tf.int32,
        name="seq_inp",
    )
    mask_inp = keras.Input(
        shape=(mbart_specs.coder_specs.seq_len,),
        dtype=tf.int8,
        name="mask_inp",
    )
    cond_inp = keras.Input(
        shape=(1,),
        dtype=tf.int32,
        name="lora_inp",
    )

    mbart_layer = MBartLikeModel(mbart_specs)

    return keras.Model(
        inputs=[seq_inp, mask_inp, cond_inp],
        outputs=mbart_layer([seq_inp, mask_inp, cond_inp]),
    )
