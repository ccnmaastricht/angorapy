#!/usr/bin/env python
"""Convolutional components/networks."""
import tensorflow as tf


# VISUAL ENCODING

def _build_visual_encoder(shape, batch_size=None, name=None):
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

@DeprecationWarning
def _build_visual_decoder(shape, batch_size=None):
    inputs = tf.keras.Input(batch_shape=(batch_size,) + shape)

    spatial_reshape_size = 7 * 7 * 64
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(spatial_reshape_size)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape([7, 7, 64])(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 3, output_padding=1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 3, output_padding=2)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(3, 5, 1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    conv_comp = _build_visual_encoder((227, 227, 3))
