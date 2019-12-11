#!/usr/bin/env python
"""TODO Module Docstring."""
import os

import gym
import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD
from tqdm import tqdm

from models.components import _build_encoding_sub_model, _build_continuous_head, _build_discrete_head
from utilities.util import env_extract_dims


def build_rnn_distinct_models(env: gym.Env, bs: int = 1):
    """Build simple policy and value models having an LSTM before their heads."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(batch_shape=(bs, None, state_dimensionality,))
    masked = tf.keras.layers.Masking()(inputs)

    # policy network
    x = TD(_build_encoding_sub_model((state_dimensionality,), bs, name="policy_encoder"), name="TD_policy")(masked)
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.SimpleRNN(32, stateful=True, return_sequences=True, batch_size=bs,
                                  activation="tanh",
                                  recurrent_initializer=tf.keras.initializers.zeros,
                                  # TODO remove debugging constraint
                                  recurrent_constraint=tf.keras.constraints.MinMaxNorm(0, 0),
                                  name="policy_recurrent_layer")(x)

    if continuous_control:
        out_policy = _build_continuous_head(n_actions, x.shape[1:], bs)(x)
    else:
        out_policy = _build_discrete_head(n_actions, x.shape[1:], bs)(x)

    # value network
    x = TD(_build_encoding_sub_model((state_dimensionality,), bs, name="value_encoder"), name="TD_value")(masked)
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.SimpleRNN(32, stateful=True, return_sequences=True, batch_size=bs,
                                  activation="tanh",
                                  recurrent_initializer=tf.keras.initializers.zeros,
                                  # TODO remove debugging constraint
                                  recurrent_constraint=tf.keras.constraints.MinMaxNorm(0, 0),
                                  name="value_recurrent_layer")(x)

    out_value = tf.keras.layers.Dense(1)(x)

    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="simple_rnn_policy")
    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="simple_rnn_value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="simple_rnn")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    environment = gym.make("LunarLanderContinuous-v2")
    pi, v, pv = build_rnn_distinct_models(environment, bs=3)

    tf.keras.utils.plot_model(pv)

    for i in tqdm(range(100000000)):
        out_pi, out_v = pv.predict(tf.random.normal((3, 16, 8)))
