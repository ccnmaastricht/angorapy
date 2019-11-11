#!/usr/bin/env python
"""Components that can be loaded into another network."""
import os

import tensorflow as tf


def build_visual_component():
    inputs = tf.keras.Input(shape=(None, 200, 200, 3))
    x = tf.keras.layers.Conv2D(32, 5, 1)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, 1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 3)(x)
    x = tf.keras.layers.Conv2D(64, 3, 3)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, 3)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(64)(x)
    outputs = tf.keras.layers.Activation("relu")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="visual_component")


def build_visual_decoder():
    spatial_reshape_size = 7 * 7 * 64

    inputs = tf.keras.Input(shape=(64,))
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(spatial_reshape_size)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Reshape([7, 7, 64])(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 3)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 3, output_padding=1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 3, output_padding=2)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2DTranspose(3, 5, 1, activation="sigmoid")(x)
    outputs = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="visual_decoder")


def build_non_visual_component(input_dimensionality: int, hidden_dimensionality: int, output_dimensionality: int):
    inputs = tf.keras.Input(shape=(input_dimensionality,))
    x = tf.keras.layers.Dense(input_dimensionality)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(hidden_dimensionality)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(output_dimensionality)(x)
    outputs = tf.keras.layers.Activation("relu")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_residual_block():
    raise NotImplementedError


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    net = build_visual_component()
    tensor = tf.random.normal([4, 200, 200, 3])
    latent = net(tensor)
    print(f"Latent Shape: {latent.shape}")

    decoder = build_visual_decoder()
    reconstruction = decoder(latent)
    print(f"Reconstruction Shape: {reconstruction.shape}")
