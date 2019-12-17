#!/usr/bin/env python
"""Collection of fully connected policy networks."""

import gym
import tensorflow as tf
from gym.spaces import Box
from tensorflow_core.python.keras.utils import plot_model

from models.components import _build_encoding_sub_model, _build_continuous_head, _build_discrete_head
from utilities.normalization import RunningNormalization
from utilities.util import env_extract_dims


def build_ffn_distinct_models(env: gym.Env):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(shape=(state_dimensionality,))
    normalized = RunningNormalization()(inputs)

    # policy network
    x = _build_encoding_sub_model(normalized.shape[1:], None, name="policy_encoder")(normalized)
    out_policy = _build_continuous_head(n_actions, (64, ), None)(x) if continuous_control \
        else _build_discrete_head(n_actions, (64, ), None)(x)
    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="policy")

    # value network
    x = _build_encoding_sub_model(normalized.shape[1:], None, name="value_encoder")(normalized)
    out_value = tf.keras.layers.Dense(1,
                                      kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                      bias_initializer=tf.keras.initializers.Constant(0.0))(x)

    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="value")

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="policy_value")


def build_ffn_shared_models(env: gym.Env):
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    # shared encoding layers
    inputs = tf.keras.Input(shape=(state_dimensionality,))
    normalized = RunningNormalization()(inputs)
    latent = _build_encoding_sub_model(normalized.shape[1:], None)(normalized)

    # policy head
    if continuous_control:
        policy_out = _build_continuous_head(n_actions, (64, ), None)(latent)
    else:
        policy_out = _build_discrete_head(n_actions, (64, ), None)(latent)
    policy = tf.keras.Model(inputs=inputs, outputs=policy_out)

    # value head
    value_out = tf.keras.layers.Dense(1, input_dim=64,
                                      kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                      bias_initializer=tf.keras.initializers.Constant(0.0))(latent)
    value = tf.keras.Model(inputs=inputs, outputs=value_out)

    return policy, value, tf.keras.Model(inputs=inputs, outputs=[policy_out, value_out])


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = "LunarLanderContinuous-v2"

    pi, vn, pv = build_ffn_distinct_models(gym.make(env))
    s_pi, s_vn, s_pv = build_ffn_shared_models(gym.make(env))

    plot_model(pv, "policy_value.png", show_shapes=True, expand_nested=True)
    plot_model(s_pv, "shared_policy_value.png", show_shapes=True, expand_nested=True)
