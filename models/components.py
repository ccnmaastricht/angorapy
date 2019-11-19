#!/usr/bin/env python
"""Components that can be loaded into another network."""
import math

import tensorflow as tf


def _build_visual_encoder(shape, batch_size=None, name=None):
    inputs = tf.keras.Input(batch_shape=(batch_size,) + shape)

    # first layer
    x = tf.keras.layers.Conv2D(96, 11, 4)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ReLU()(x)

    # second layer
    x = tf.keras.layers.Conv2D(256, 5, 1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ReLU()(x)

    # third layer
    x = tf.keras.layers.Conv2D(384, 3, 1)(x)
    x = tf.keras.layers.ReLU()(x)

    # fourth layer
    x = tf.keras.layers.Conv2D(384, 3, 1)(x)
    x = tf.keras.layers.ReLU()(x)

    # fifth layer
    x = tf.keras.layers.Conv2D(256, 3, 1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ReLU()(x)

    # fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(4096)(x)
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


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
    x = tf.keras.layers.Conv2DTranspose(3, 5, 1, activation="sigmoid")(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def _build_non_visual_component(input_dim: int, hidden_dim: int, output_dim: int, batch_size: int = None):
    inputs = tf.keras.Input(batch_shape=(batch_size, input_dim))

    x = tf.keras.layers.Dense(hidden_dim)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def _build_encoding_sub_model(inputs):
    x = tf.keras.layers.Dense(64, kernel_initializer=DENSE_INIT)(inputs)
    x = tf.keras.layers.ReLU("tanh")(x)
    x = tf.keras.layers.Dense(64, kernel_initializer=DENSE_INIT)(x)

    return tf.keras.layers.Activation("tanh")(x)


def _build_continuous_head(n_actions, inputs):
    means = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(inputs)
    means = tf.keras.layers.Activation("linear")(means)

    stdevs = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(inputs)
    stdevs = tf.keras.layers.Activation("softplus")(stdevs)

    return tf.keras.layers.Concatenate()([means, stdevs])


def _build_discrete_head(n_actions, inputs):
    x = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(inputs)
    return tf.keras.layers.Activation("softmax")(x)


DENSE_INIT = tf.keras.initializers.orthogonal(gain=math.sqrt(2))
