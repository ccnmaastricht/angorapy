#!/usr/bin/env python
"""Components that can be loaded into another network."""
import math

import tensorflow as tf


# ABSTRACTION COMPONENTS
from models.layers import StdevLayer, StdevLayer


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

    # inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))
    #
    # x = tf.keras.layers.Dense(layer_sizes[0])(inputs)
    # x = tf.keras.layers.Activation("tanh")(x)
    #
    # for i in range(1, len(layer_sizes)):
    #     x = tf.keras.layers.Dense(layer_sizes[i])(x)
    #     x = tf.keras.layers.Activation("tanh")(x)

    model = tf.keras.Sequential(name=name)
    for i in range(len(layer_sizes)):
        model.add(tf.keras.layers.Dense(layer_sizes[i],
                                        kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.0))))
        model.add(tf.keras.layers.Activation("tanh"))
    model.build(input_shape=(batch_size, ) + shape)

    # return tf.keras.Model(inputs=inputs, outputs=x, name=name)
    return model

# POLICY HEADS


def _build_continuous_head(n_actions, input_shape, batch_size, stdevs_from_latent=False):
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(input_shape))
    means = tf.keras.layers.Dense(n_actions, name="means",
                                  kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                  bias_initializer=tf.keras.initializers.Constant(0.0))(inputs)
    if stdevs_from_latent:
        stdevs = tf.keras.layers.Dense(n_actions, name="log_stdevs")(inputs)
    else:
        stdevs = StdevLayer(n_actions, name="log_stdevs")(means)

    return tf.keras.Model(inputs=inputs, outputs=[means, stdevs], name="continuous_action_head")


def _build_discrete_head(n_actions, input_shape, batch_size):
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(input_shape))

    x = tf.keras.layers.Dense(n_actions,
                              kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                              bias_initializer=tf.keras.initializers.Constant(0.0))(inputs)
    x = tf.nn.log_softmax(x, name="log_likelihoods")

    return tf.keras.Model(inputs=inputs, outputs=x, name="discrete_action_head")
