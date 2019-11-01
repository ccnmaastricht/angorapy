#!/usr/bin/env python
"""Collection of fully connected policy networks."""
import gym
import numpy
import tensorflow as tf
from gym.spaces import Box

from utilities.util import env_extract_dims


def build_ffn_actor_model(env: gym.Env):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(shape=(state_dimensionality,))
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    if continuous_control:
        means = tf.keras.layers.Dense(n_actions, input_dim=64)(x)
        means = tf.keras.layers.Activation("linear")(means)
        stdevs = tf.keras.layers.Dense(n_actions, input_dim=64)(x)
        stdevs = tf.keras.layers.Activation("softplus")(stdevs)

        out = tf.keras.layers.Concatenate()([means, stdevs])
    else:
        x = tf.keras.layers.Dense(n_actions)(x)
        out = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=out)


def build_ffn_critic_model(env: gym.Env):
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(shape=(state_dimensionality,))
    x = tf.keras.layers.Dense(64, input_dim=state_dimensionality)(inputs)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.Dense(64, input_dim=64)(x)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.Dense(1, input_dim=64)(x)
    out = tf.keras.layers.Activation("linear")(x)

    return tf.keras.Model(inputs=inputs, outputs=out)
