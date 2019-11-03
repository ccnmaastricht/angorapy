#!/usr/bin/env python
"""Collection of fully connected policy networks."""
import math

import gym
import tensorflow as tf
from gym.spaces import Box
from tensorflow_core.python.keras.utils import plot_model

from utilities.util import env_extract_dims


DENSE_INIT = tf.keras.initializers.orthogonal(gain=math.sqrt(2))


def _build_encoding_sub_model(input_size, name: str = None):
    inputs = tf.keras.Input(shape=(input_size,))
    x = tf.keras.layers.Dense(64, kernel_initializer=DENSE_INIT)(inputs)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.Dense(64, kernel_initializer=DENSE_INIT)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def build_ffn_distinct_models(env: gym.Env):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(shape=(state_dimensionality,))

    # policy network
    policy_latent = _build_encoding_sub_model(state_dimensionality)
    x = policy_latent(inputs)
    if continuous_control:
        means = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(x)
        means = tf.keras.layers.Activation("linear")(means)
        stdevs = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(x)
        stdevs = tf.keras.layers.Activation("softplus")(stdevs)

        out_policy = tf.keras.layers.Concatenate()([means, stdevs])
    else:
        x = tf.keras.layers.Dense(n_actions, kernel_initializer=DENSE_INIT)(x)
        out_policy = tf.keras.layers.Activation("softmax")(x)

    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="policy")

    # value network
    value_latent = _build_encoding_sub_model(state_dimensionality)
    x = value_latent(inputs)
    x = tf.keras.layers.Dense(1, input_dim=64)(x)
    out_value = tf.keras.layers.Activation("linear")(x)

    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="policy_value")


def build_ffn_shared_models(env: gym.Env):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    # shared encoding layers
    inputs = tf.keras.Input(shape=(state_dimensionality,))
    latent = _build_encoding_sub_model(state_dimensionality, name="Encoder")
    latent_representation = latent(inputs)

    # policy head
    if continuous_control:
        means = tf.keras.layers.Dense(n_actions)(latent_representation)
        means = tf.keras.layers.Activation("linear")(means)
        stdevs = tf.keras.layers.Dense(n_actions)(x)
        stdevs = tf.keras.layers.Activation("softplus")(stdevs)

        policy_out = tf.keras.layers.Concatenate()([means, stdevs])
    else:
        x = tf.keras.layers.Dense(n_actions)(latent_representation)
        policy_out = tf.keras.layers.Activation("softmax")(x)
    policy = tf.keras.Model(inputs=inputs, outputs=policy_out)

    # value head
    x = tf.keras.layers.Dense(1, input_dim=64)(latent_representation)
    value_out = tf.keras.layers.Activation("linear")(x)
    value = tf.keras.Model(inputs=inputs, outputs=value_out)

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[policy_out, value_out])


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    policy, value, policy_value = build_ffn_distinct_models(gym.make("CartPole-v1"))
    s_policy, s_value, s_policy_value = build_ffn_shared_models(gym.make("CartPole-v1"))
    plot_model(policy_value, "policy_value.png", show_shapes=True)
    plot_model(s_policy_value, "shared_policy_value.png", show_shapes=True)
