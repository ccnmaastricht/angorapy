#!/usr/bin/env python
"""Collection of fully connected policy networks."""
import math

import gym
import tensorflow as tf
from gym.spaces import Box
from tensorflow_core.python.keras.utils import plot_model

from utilities.normalization import RunningNormalization
from utilities.util import env_extract_dims


DENSE_INIT = tf.keras.initializers.orthogonal(gain=math.sqrt(2))


def _build_encoding_sub_model(inputs):
    x = tf.keras.layers.Dense(64, kernel_initializer=DENSE_INIT)(inputs)
    x = tf.keras.layers.Activation("tanh")(x)
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


def build_ffn_distinct_models(env: gym.Env):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(shape=(state_dimensionality,))

    # policy network
    normalized = RunningNormalization()(inputs)
    x = _build_encoding_sub_model(normalized)
    out_policy = _build_continuous_head(n_actions, x) if continuous_control else _build_discrete_head(n_actions, x)
    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="policy")

    # value network
    normalized = RunningNormalization()(inputs)
    x = _build_encoding_sub_model(normalized)
    x = tf.keras.layers.Dense(1, input_dim=64)(x)
    out_value = tf.keras.layers.Activation("linear")(x)

    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="policy_value")


def build_ffn_shared_models(env: gym.Env):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    # shared encoding layers
    inputs = tf.keras.Input(shape=(state_dimensionality,))
    normalized = RunningNormalization()(inputs)
    latent = _build_encoding_sub_model(normalized)

    # policy head
    if continuous_control:
        means = tf.keras.layers.Dense(n_actions)(latent)
        means = tf.keras.layers.Activation("linear")(means)
        stdevs = tf.keras.layers.Dense(n_actions)(latent)
        stdevs = tf.keras.layers.Activation("softplus")(stdevs)

        policy_out = tf.keras.layers.Concatenate()([means, stdevs])
    else:
        x = tf.keras.layers.Dense(n_actions)(latent)
        policy_out = tf.keras.layers.Activation("softmax")(x)
    policy = tf.keras.Model(inputs=inputs, outputs=policy_out)

    # value head
    x = tf.keras.layers.Dense(1, input_dim=64)(latent)
    value_out = tf.keras.layers.Activation("linear")(x)
    value = tf.keras.Model(inputs=inputs, outputs=value_out)

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[policy_out, value_out])


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = "LunarLanderContinuous-v2"

    pi, vn, pv = build_ffn_distinct_models(gym.make(env))
    s_pi, s_vn, s_pv = build_ffn_shared_models(gym.make(env))
    pi.summary()

    plot_model(pv, "policy_value.png", show_shapes=True)
    plot_model(s_pv, "shared_policy_value.png", show_shapes=True)
