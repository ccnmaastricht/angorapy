#!/usr/bin/env python
"""TODO Module Docstring."""
import os

import gym
from gym.spaces import Box
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as TD

from models.components import _build_encoding_sub_model, _build_continuous_head, _build_discrete_head
from utilities.util import env_extract_dims


def build_rnn_distinct_models(env: gym.Env, bs: int):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(batch_shape=(bs, None, state_dimensionality,))

    # policy network
    x = TD(_build_encoding_sub_model((state_dimensionality, ), bs), name="TD_policy")(inputs)
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.LSTM(32, stateful=True)(x)

    out_policy = _build_continuous_head(n_actions, (32, ), bs)(x) if continuous_control \
        else _build_discrete_head(n_actions, (32, ), bs)(x)
    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="policy")

    # value network
    x = TD(_build_encoding_sub_model((state_dimensionality, ), bs), name="TD_value")(inputs)
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, batch_size=bs)(x)

    x = tf.keras.layers.Dense(1)(x)
    out_value = tf.keras.layers.Activation("linear")(x)

    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="policy_value")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    environment = gym.make("LunarLanderContinuous-v2")
    pi, v, pv = build_rnn_distinct_models(environment, bs=1)

    tf.keras.utils.plot_model(pv)
