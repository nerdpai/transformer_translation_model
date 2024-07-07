from tensorflow._api.v2.v2 import keras

from module_5.changeable import SEED

WEIGHTS_INITIALIZER = keras.initializers.GlorotUniform(seed=SEED)
BIAS_INITIALIZER = keras.initializers.RandomUniform(seed=SEED)
EMBEDDING_INITIALIZER = keras.initializers.RandomUniform(seed=SEED)
DIM_BOOM_FACTOR = 4
NORM_EPSILON = 1e-5
