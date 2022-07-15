#!/usr/bin/env python
"""Components that can be loaded into another network."""

import tensorflow as tf


def _build_fcn_component(input_dim: int, hidden_dim: int, output_dim: int, batch_size: int = None,
                         name: str = None):
    inputs = tf.keras.Input(batch_shape=(batch_size, input_dim))

    x = tf.keras.layers.Dense(hidden_dim)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def _build_encoding_sub_model(shape, batch_size, layer_sizes=(64, 64), name=None):
    assert len(layer_sizes) > 0, "You need at least one layer in an encoding submodel."

    inputs = tf.keras.Input(shape=shape, batch_size=batch_size, name=f"{name}_input")

    x = inputs
    for i in range(len(layer_sizes)):
        x = tf.keras.layers.Dense(layer_sizes[i],
                                  kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.0)),
                                  bias_initializer=tf.constant_initializer(0.0),
                                  name=f"{name}_{i}")(x)
        x = tf.keras.layers.Activation("tanh")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
