#!/usr/bin/env python
"""Collection of generic fully connected and recurrent policy networks."""

import os
from typing import Tuple, Iterable

import gym
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.python.keras.utils.vis_utils import plot_model

from agent.policies import BasePolicyDistribution, CategoricalPolicyDistribution, GaussianPolicyDistribution, \
    BetaPolicyDistribution
from models.components import _build_encoding_sub_model
from utilities.util import env_extract_dims


def build_ffn_models(env: gym.Env, distribution: BasePolicyDistribution, shared: bool = False,
                     layer_sizes: Tuple = (64, 64)):
    """Build simple two-layer model."""

    # preparation
    state_dimensionality, n_actions = env_extract_dims(env)

    # input preprocessing
    inputs = tf.keras.Input(shape=(state_dimensionality,))

    # policy network
    latent = _build_encoding_sub_model(inputs.shape[1:], None, layer_sizes=layer_sizes,
                                       name="policy_encoder")(inputs)
    out_policy = distribution.build_action_head(n_actions, (layer_sizes[-1],), None)(latent)

    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="policy")

    # value network
    if not shared:
        value_latent = _build_encoding_sub_model(inputs.shape[1:], None, layer_sizes=layer_sizes,
                                                 name="value_encoder")(inputs)
        value_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(value_latent)
    else:
        value_out = tf.keras.layers.Dense(1, input_dim=layer_sizes[-1], kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(latent)

    value = tf.keras.Model(inputs=inputs, outputs=value_out, name="value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, value_out], name="policy_value")


def build_rnn_models(env: gym.Env, distribution: BasePolicyDistribution, shared: bool = False, bs: int = 1,
                     model_type: str = "rnn", layer_sizes: Tuple = (64, )):
    """Build simple policy and value models having a recurrent layer before their heads."""
    state_dimensionality, n_actions = env_extract_dims(env)
    rnn_choice = {"rnn": tf.keras.layers.SimpleRNN,
                  "lstm": tf.keras.layers.LSTM,
                  "gru": tf.keras.layers.GRU}[
        model_type]

    sequence_length = None

    inputs = tf.keras.Input(batch_shape=(bs, sequence_length, state_dimensionality,))
    masked = tf.keras.layers.Masking(batch_input_shape=(bs, sequence_length, state_dimensionality,))(inputs)

    # policy network; stateful, so batch size needs to be known
    x = TD(_build_encoding_sub_model((state_dimensionality,), bs, layer_sizes=layer_sizes, name="policy_encoder"),
           name="TD_policy")(masked)
    x.set_shape([bs] + x.shape[1:])

    x, *_ = rnn_choice(layer_sizes[-1],
                       stateful=True,
                       return_sequences=True,
                       return_state=True,
                       batch_size=bs,
                       name="policy_recurrent_layer")(x)

    out_policy = distribution.build_action_head(n_actions, x.shape[1:], bs)(x)

    # value network
    if not shared:
        x = TD(_build_encoding_sub_model((state_dimensionality,), bs, layer_sizes=layer_sizes, name="value_encoder"),
               name="TD_value")(masked)
        x.set_shape([bs] + x.shape[1:])
        x, *_ = rnn_choice(layer_sizes[-1], stateful=True, return_sequences=True, return_state=True, batch_size=bs,
                           name="value_recurrent_layer")(x)
        out_value = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(x)
    else:
        out_value = tf.keras.layers.Dense(1, input_dim=x.shape[1:], kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(x)

    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="simple_rnn_policy")
    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="simple_rnn_value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="simple_rnn")


def build_simple_models(env: gym.Env, distribution: BasePolicyDistribution, shared: bool = False, bs: int = 1,
                        model_type: str = "rnn", **kwargs):
    """Build simple networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for ffn and rnns
    if model_type == "ffn":
        return build_ffn_models(env, distribution, shared)
    else:
        return build_rnn_models(env, distribution, shared, bs=bs, model_type=model_type)


def build_deeper_models(env: gym.Env, distribution: BasePolicyDistribution, shared: bool = False, bs: int = 1,
                        model_type: str = "rnn", **kwargs):
    """Build deeper simple networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for ffn and rnns
    if model_type == "ffn":
        return build_ffn_models(env, distribution, shared, layer_sizes=(64, 64, 64, 32))
    else:
        return build_rnn_models(env, distribution, shared, bs=bs, model_type=model_type, layer_sizes=(64, 64, 64, 32))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    cont_env = gym.make("LunarLanderContinuous-v2")
    disc_env = gym.make("LunarLander-v2")

    # _, _, ffn_distinct = build_ffn_models(cont_env, GaussianPolicyDistribution(cont_env), False)
    # _, _, ffn_shared = build_ffn_models(cont_env, BetaPolicyDistribution(cont_env), True)
    # _, _, ffn_distinct_discrete = build_ffn_models(disc_env, CategoricalPolicyDistribution(disc_env), True)
    _, _, rnn_distinct = build_rnn_models(cont_env, BetaPolicyDistribution(cont_env), False, 1, "lstm")
    # _, _, rnn_shared = build_rnn_models(cont_env, GaussianPolicyDistribution(cont_env), True, 1, "gru")
    # _, _, rnn_shared_discrete = build_rnn_models(disc_env, CategoricalPolicyDistribution(disc_env), True)

    # plot_model(ffn_distinct, "ffn_distinct.png", expand_nested=True)
    # plot_model(ffn_distinct_discrete, "ffn_distinct_discrete.png", expand_nested=True)
    # plot_model(ffn_shared, "ffn_shared.png", expand_nested=True)
    # plot_model(rnn_distinct, "rnn_distinct.png", expand_nested=True)
    # plot_model(rnn_shared, "rnn_shared.png", expand_nested=True)
    # plot_model(rnn_shared_discrete, "rnn_shared_discrete.png", expand_nested=True)
