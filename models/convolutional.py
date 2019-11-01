#!/usr/bin/env python
"""Convolutional Networks serving as policy or critic for agents getting visual input."""
import tensorflow as tf
from gym.spaces import Box

from utilities.util import env_extract_dims


def build_cnn_actor_model(env):
    # TODO add support for continuous action space and variable frame shape
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(shape=(30, 30, 1))

    # convolutions
    x = tf.keras.layers.Conv2D(32, 8, 4, input_shape=(30, 30, 1))(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, 2)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.ZeroPadding2D(1, data_format="channels_last")(x)
    x = tf.keras.layers.Conv2D(128, 3, 1)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # fully connected
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(n_actions, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=out)


def build_cnn_critic_model(env):
    # TODO add support for continuous action space and variable frame shape
    # TODO way too much duplicate code, should solve this more cleverly
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(shape=(30, 30, 1))

    # convolutions
    x = tf.keras.layers.Conv2D(32, 8, 4, input_shape=(30, 30, 1))(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, 2)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.ZeroPadding2D(1, data_format="channels_last")(x)
    x = tf.keras.layers.Conv2D(128, 3, 1)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # fully connected
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(1, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=out)
