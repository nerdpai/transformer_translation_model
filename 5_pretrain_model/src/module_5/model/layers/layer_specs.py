import numpy
import tensorflow._api.v2.v2 as tf
from typing import TypeAlias, Any, Union
from dataclasses import dataclass


DTypeLike: TypeAlias = tf.DType | str | numpy.dtype[Any] | int


@dataclass(frozen=True)
class LayerSpecs:
    trainable: bool = True
    name: Union[str, None] = None
    dtype: DTypeLike = tf.float32
    dynamic: bool = False
