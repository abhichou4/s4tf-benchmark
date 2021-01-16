import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def DenseModel():

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation="relu", name="layer1"))
    model.add(layers.Dense(10, activation="relu", name="layer2"))
    model.add(layers.Dense(10, name="layer3"))
    return model
