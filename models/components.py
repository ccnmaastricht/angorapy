#!/usr/bin/env python
"""Components that can be loaded into another network."""
import os

import tensorflow as tf
from tensorflow import keras


def build_visual_component():
    inputs = keras.Input(shape=(200, 200, 3))
    x = keras.layers.Conv2D(32, 5, 1, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, 1, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 3)(x)
    x = tf.keras.layers.Conv2D(64, 3, 3, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, 3, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(64, activation="relu")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def build_visual_decoder():
    spatial_reshape_size = 7 * 7 * 64

    inputs = keras.Input(shape=(64,))

    # TODO spatial softmax
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(spatial_reshape_size, activation="relu")(x)
    x = tf.keras.layers.Reshape([7, 7, 64])(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 3, activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 3, activation="relu", output_padding=1)(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 3, activation="relu", output_padding=2)(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 1, activation="relu")(x)
    outputs = tf.keras.layers.Conv2DTranspose(3, 5, 1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def build_non_visual_component(input_dimensionality: int, hidden_dimensionality: int, output_dimensionality: int):
    inputs = keras.Input(shape=(input_dimensionality,))
    x = tf.keras.layers.Dense(input_dimensionality, activation="relu")(inputs)
    x = tf.keras.layers.Dense(hidden_dimensionality, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dimensionality, activation="relu")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


class ResNetBlock(tf.keras.Model):
    """ResNet Block with variable number of convolutional sub-layers."""

    def __init__(self, filters, kernel_size, stride):
        super(ResNetBlock, self).__init__(name='')

        self.forward = tf.keras.Sequential()

        for n_filters in filters:
            self.forward.add(tf.keras.layers.Conv2D(n_filters, kernel_size, stride, padding="same"))
            self.forward.add(tf.keras.layers.BatchNormalization())
            self.forward.add(tf.keras.layers.ReLU())

    def call(self, input_tensor, training=False, **kwargs):
        x = self.forward(input_tensor)

        x += input_tensor
        return tf.nn.relu(x)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    net = VisualComponent()
    tensor = tf.random.normal([4, 200, 200, 3])
    latent = net(tensor)
    print(f"Latent Shape: {latent.shape}")

    decoder = VisualDecoder()
    reconstruction = decoder(latent)
    print(f"Reconstruction Shape: {reconstruction.shape}")
