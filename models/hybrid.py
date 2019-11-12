#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time

import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD

from environments import *
from models.components import build_visual_component, build_non_visual_component
from utilities.util import env_extract_dims


def build_shadow_brain(env: gym.Env):
    """Build network for the shadow hand task."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32
    print(n_actions)

    # inputs
    visual_input = tf.keras.Input(shape=(200, 200, 3), name="visual_input")
    proprioceptive_input = tf.keras.Input(shape=(48,), name="proprioceptive_input")
    somatosensory_input = tf.keras.Input(shape=(6,), name="somatosensory_input")
    goal_input = tf.keras.Input(shape=(7,), name="goal_input")
    hidden_state_input = tf.keras.Input(shape=(hidden_dimensions,))

    visual_abstraction = build_visual_component()(visual_input)
    proprioceptive_abstraction = build_non_visual_component(48, 12, 8)(proprioceptive_input)
    somatosensory_abstraction = build_non_visual_component(6, 24, 8)(somatosensory_input)

    concatenation = tf.keras.layers.Concatenate()([visual_abstraction,
                                                   proprioceptive_abstraction,
                                                   somatosensory_abstraction])

    x = tf.keras.layers.Dense(48)(concatenation)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Concatenate()([x, goal_input])

    x, h = tf.keras.layers.LSTMCell(hidden_dimensions)(x, hidden_state_input)

    policy_outputs = tf.keras.layers.Dense(n_actions)(x)
    value_outputs = tf.keras.layers.Dense(1)(x)

    # models
    policy = tf.keras.Model(inputs=[visual_input, proprioceptive_input, somatosensory_input, goal_input],
                            outputs=[policy_outputs, h])
    value = tf.keras.Model(inputs=[visual_input, proprioceptive_input, somatosensory_input, goal_input],
                           outputs=[value_outputs, h])
    policy_value = tf.keras.Model(inputs=[visual_input, proprioceptive_input, somatosensory_input, goal_input],
                                  outputs=[policy_outputs, value_outputs, h])

    return policy, value, policy_value


def init_hidden(shape):
    return tf.zeros(shape=shape)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sequence_length = 10
    batch_size = 16

    policy, value, policy_value = build_shadow_brain(gym.make("ShadowHand-v0"))
    tf.keras.utils.plot_model(policy, show_shapes=True, expand_nested=True)

    start_time = time.time()
    hidden = init_hidden(shape=(16, 32))
    for t in range(sequence_length):
        input_data = (tf.random.normal([batch_size, 200, 200, 3]),
                      tf.random.normal([batch_size, 24]),
                      tf.random.normal([batch_size, 32]),
                      tf.random.normal([batch_size, 4]),
                      hidden)

        out, hidden = policy(input_data)

    print(f"Execution Time: {time.time() - start_time}")
