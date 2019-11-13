#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time

import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD

from environments import *
from models.components import build_visual_component, build_non_visual_component
from models.fully_connected import _build_continuous_head, _build_discrete_head
from utilities.util import env_extract_dims


def build_shadow_brain(env: gym.Env):
    """Build network for the shadow hand task."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32

    # inputs
    visual_in = tf.keras.Input(shape=(None, 200, 200, 3), name="visual_input")
    proprio_in = tf.keras.Input(shape=(None, 48,), name="proprioceptive_input")
    touch_in = tf.keras.Input(shape=(None, 92,), name="somatosensory_input")
    goal_in = tf.keras.Input(shape=(None, 7,), name="goal_input")

    visual_abstraction = TD(build_visual_component())(visual_in)
    proprioceptive_abstraction = TD(build_non_visual_component(48, 12, 8))(proprio_in)
    somatosensory_abstraction = TD(build_non_visual_component(92, 24, 8))(touch_in)

    concatenation = tf.keras.layers.Concatenate()([visual_abstraction,
                                                   proprioceptive_abstraction,
                                                   somatosensory_abstraction])

    x = TD(tf.keras.layers.Dense(48))(concatenation)
    x = TD(tf.keras.layers.Dense(32))(x)
    x = tf.keras.layers.Concatenate()([x, goal_in])
    x = tf.keras.layers.LSTM(hidden_dimensions)(x)

    policy_out = _build_continuous_head(n_actions, x) if continuous_control else _build_discrete_head(n_actions, x)
    value_out = tf.keras.layers.Dense(1)(x)

    # models
    policy = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in], outputs=[policy_out])
    value = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in], outputs=[value_out])
    policy_value = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in], outputs=[policy_out, value_out])

    return policy, value, policy_value


def init_hidden(shape):
    """Get initial hidden state"""
    return tf.zeros(shape=shape)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sequence_length = 10
    batch_size = 16

    pi, vn, pv = build_shadow_brain(gym.make("ShadowHand-v1"))
    tf.keras.utils.plot_model(pi, show_shapes=True, expand_nested=True)

    start_time = time.time()
    for t in range(sequence_length):
        input_data = (tf.random.normal([batch_size, 1, 200, 200, 3]),
                      tf.random.normal([batch_size, 1, 48]),
                      tf.random.normal([batch_size, 1, 92]),
                      tf.random.normal([batch_size, 1, 7]))

        out = pi(input_data)

    print(f"Execution Time: {time.time() - start_time}")
