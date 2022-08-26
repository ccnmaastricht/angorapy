from typing import List

import tensorflow as tf


def build_encoder(layer_sizes: List[int]):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation="sigmoid") for size in layer_sizes
    ])

    return encoder


def build_decoder(layer_sizes: List[int]):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation="sigmoid") for size in layer_sizes
    ])

    return encoder