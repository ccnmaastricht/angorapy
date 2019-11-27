#!/usr/bin/env python
"""Components that can be loaded into another network."""
import math

import tensorflow as tf


# ABSTRACTION COMPONENTS

def _build_fcn_component(input_dim: int, hidden_dim: int, output_dim: int, batch_size: int = None,
                         name: str = None):
    inputs = tf.keras.Input(batch_shape=(batch_size, input_dim))

    x = tf.keras.layers.Dense(hidden_dim)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def _build_encoding_sub_model(shape, batch_size):
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))

    x = tf.keras.layers.Dense(64, kernel_initializer=DENSE_INIT)(inputs)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.Dense(64, kernel_initializer=DENSE_INIT)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# POLICY HEADS

def _build_continuous_head(n_actions, input_shape, batch_size):
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(input_shape))

    means = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT, name="means")(inputs)

    stdevs = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(inputs)
    stdevs = tf.keras.layers.Activation("softplus", name="stdevs")(stdevs)

    concat = tf.keras.layers.Concatenate(name="multivariates")([means, stdevs])

    return tf.keras.Model(inputs=inputs, outputs=concat)


def _build_discrete_head(n_actions, input_shape, batch_size):
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(input_shape))

    x = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(inputs)
    x = tf.keras.layers.Activation("softmax", name="actions")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


DENSE_INIT = tf.keras.initializers.orthogonal(gain=math.sqrt(2))
