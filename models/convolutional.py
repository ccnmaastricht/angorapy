#!/usr/bin/env python
"""Convolutional components/networks."""

import tensorflow as tf


# VISUAL ENCODING

def _build_visual_encoder(shape, batch_size=None, name="visual_component"):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory."""
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))

    # first layer
    x = tf.keras.layers.Conv2D(32, 11, 4)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)

    # second layer
    x = tf.keras.layers.Conv2D(32, 5, 1, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(3, 2)(x)

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

    # fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


# VISUAL DECODING

def _build_visual_decoder(input_dim):
    model = tf.keras.Sequential((
        tf.keras.layers.Dense(512, input_dim=input_dim, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1152, activation="relu"),

        tf.keras.layers.Reshape((6, 6, 32)),
        tf.keras.layers.Conv2DTranspose(32, 3, 2),
        tf.keras.layers.Conv2DTranspose(64, 3, 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(64, 3, 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(32, 3, 2),
        tf.keras.layers.Conv2DTranspose(32, 5, 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(32, 3, 2),
        tf.keras.layers.Conv2DTranspose(3, 11, 4, activation="sigmoid"),
    ))

    model.build()
    return model


if __name__ == "__main__":
    conv_comp = _build_visual_encoder((227, 227, 3))
    conv_dec = _build_visual_decoder(512)
    conv_comp.summary()
    conv_dec.summary()
