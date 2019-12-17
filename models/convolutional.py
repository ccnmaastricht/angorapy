#!/usr/bin/env python
"""Convolutional components/networks."""
import math

import numpy
import tensorflow as tf


# VISUAL ENCODING

def _build_visual_encoder(shape, batch_size=None, name="visual_component"):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory."""
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))

    # first layer
    x = tf.keras.layers.Conv2D(32, 11, 4)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ReLU()(x)

    # second layer
    x = tf.keras.layers.Conv2D(32, 5, 1, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ReLU()(x)

    # third layer
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    # fourth layer
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    # fifth layer
    x = tf.keras.layers.Conv2D(32, 3, 1, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ReLU()(x)

    # fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


# VISUAL DECODING

def _build_visual_decoder(input_dim, spatial_dims):
    inputs = tf.keras.Input((input_dim,))

    x = tf.keras.layers.Dense(input_dim * 2, activation="tanh")(inputs)
    x = tf.keras.layers.Dense(input_dim * 4, activation="tanh")(x)
    x = tf.keras.layers.Dense(input_dim * 4, activation="tanh")(x)
    x = tf.keras.layers.Dense(numpy.prod(spatial_dims) * 3, activation="sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    conv_comp = _build_visual_encoder((227, 227, 3))
