import tensorflow._api.v2.v2 as tf
from abc import ABC, abstractmethod
from typing import Any


class TrainableTest(ABC):
    @abstractmethod
    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def prepare_layers(self) -> None:
        pass

    @abstractmethod
    def prepare_model(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> Any:
        pass

    def _get_repeat_tensor(self, tensor: tf.Tensor, r: int) -> tf.Tensor:
        return tf.repeat(tensor, r, axis=0)

    def _before_after_text(self, name: str, before: Any, after: Any) -> str:
        return (
            f"{name} performs better before training\n"
            + f"before: {before}\n"
            + f"after: {after}\n"
            + "check this inconvenience"
        )
