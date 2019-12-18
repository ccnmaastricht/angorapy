#!/usr/bin/env python
"""Collection of generic fully connected and recurrent policy networks."""

import os

import gym
import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow_core.python.keras.utils import plot_model

from models.components import _build_encoding_sub_model, _build_continuous_head, _build_discrete_head
from utilities.normalization import RunningNormalization
from utilities.util import env_extract_dims


def build_ffn_models(env: gym.Env, shared: bool = False, **kwargs):
    # preparation
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)
    encoding_layer_sizes = (64, 64)

    # input preprocessing
    inputs = tf.keras.Input(shape=(state_dimensionality,))
    normalized = RunningNormalization()(inputs)

    # policy network
    latent = _build_encoding_sub_model(normalized.shape[1:], None, layer_sizes=encoding_layer_sizes,
                                       name="policy_encoder")(normalized)
    if continuous_control:
        out_policy = _build_continuous_head(n_actions, (64,), None)(latent)
    else:
        out_policy = _build_discrete_head(n_actions, (64,), None)(latent)

    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="policy")

    # value network
    if not shared:
        value_latent = _build_encoding_sub_model(normalized.shape[1:], None, layer_sizes=encoding_layer_sizes,
                                                 name="value_encoder")(normalized)
        value_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(value_latent)
    else:
        value_out = tf.keras.layers.Dense(1, input_dim=64, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(latent)

    value = tf.keras.Model(inputs=inputs, outputs=value_out, name="value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, value_out], name="policy_value")


def build_rnn_models(env: gym.Env, bs: int = 1, shared: bool = False, model_type: str = "rnn"):
    """Build simple policy and value models having an LSTM before their heads."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)
    RNNChoice = {"rnn": tf.keras.layers.SimpleRNN, "lstm": tf.keras.layers.LSTM, "gru": tf.keras.layers.GRU}[model_type]

    inputs = tf.keras.Input(batch_shape=(bs, None, state_dimensionality,))
    masked = tf.keras.layers.Masking()(inputs)

    # policy network
    x = TD(_build_encoding_sub_model((state_dimensionality,), bs, layer_sizes=(64,), name="policy_encoder"),
           name="TD_policy")(masked)
    x.set_shape([bs] + x.shape[1:])
    x = RNNChoice(64, stateful=True, return_sequences=True, batch_size=bs, name="policy_recurrent_layer")(x)

    if continuous_control:
        out_policy = _build_continuous_head(n_actions, x.shape[1:], bs)(x)
    else:
        out_policy = _build_discrete_head(n_actions, x.shape[1:], bs)(x)

    # value network
    if not shared:
        x = TD(_build_encoding_sub_model((state_dimensionality,), bs, layer_sizes=(64,), name="value_encoder"),
               name="TD_value")(masked)
        x.set_shape([bs] + x.shape[1:])
        x = RNNChoice(64, stateful=True, return_sequences=True, batch_size=bs, name="value_recurrent_layer")(x)
        out_value = tf.keras.layers.Dense(1)(x)
    else:
        out_value = tf.keras.layers.Dense(1, input_dim=64, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(x)

    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="simple_rnn_policy")
    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="simple_rnn_value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="simple_rnn")


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    _, _, ffn_distinct = build_ffn_models(gym.make("LunarLanderContinuous-v2"), False)
    _, _, ffn_shared = build_ffn_models(gym.make("LunarLanderContinuous-v2"), True)
    _, _, rnn_distinct = build_rnn_models(gym.make("LunarLanderContinuous-v2"), 1, False, "lstm")
    _, _, rnn_shared = build_rnn_models(gym.make("LunarLanderContinuous-v2"), 1, True, "gru")

    plot_model(ffn_distinct, "ffn_distinct.png")
    plot_model(ffn_shared, "ffn_shared.png")
    plot_model(rnn_distinct, "rnn_distinct.png")
    plot_model(rnn_shared, "rnn_shared.png")
