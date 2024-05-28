import os
import random
import numpy as np
import tensorflow as tf
from keras import utils


def execute(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    utils.set_random_seed(seed)
