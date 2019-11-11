#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time

import gym
import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TimeDist

from environments import *
from models.components import build_visual_component, build_non_visual_component
from utilities.util import env_extract_dims


def build_shadow_brain(env: gym.Env):
    """Build network for the shadow hand task."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    visual_input = tf.keras.Input(shape=(None, 200, 200, 3), name="visual_input")
    proprioceptive_input = tf.keras.Input(shape=(None, 48,), name="proprioceptive_input")
    somatosensory_input = tf.keras.Input(shape=(None, 6,), name="somatosensory_input")
    goal_input = tf.keras.Input(shape=(None, 7,), name="goal_input")

    visual_abstraction = TimeDist(build_visual_component())(visual_input)
    proprioceptive_abstraction = TimeDist(build_non_visual_component(48, 12, 8))(
        proprioceptive_input)
    somatosensory_abstraction = TimeDist(build_non_visual_component(6, 24, 8))(
        somatosensory_input)

    concatenation = tf.keras.layers.Concatenate()([visual_abstraction,
                                                   proprioceptive_abstraction,
                                                   somatosensory_abstraction])

    x = TimeDist(tf.keras.layers.Dense(48))(concatenation)
    x = TimeDist(tf.keras.layers.Dense(32))(x)
    x = tf.keras.layers.Concatenate()([x, goal_input])

    x = tf.keras.layers.LSTM(32, )(x)

    policy_outputs = tf.keras.layers.Dense(n_actions)(x)
    value_outputs = tf.keras.layers.Dense(1)(x)

    # models
    policy = tf.keras.Model(inputs=[visual_input, proprioceptive_input, somatosensory_input, goal_input],
                            outputs=policy_outputs)
    value = tf.keras.Model(inputs=[visual_input, proprioceptive_input, somatosensory_input, goal_input],
                           outputs=value_outputs)
    policy_value = tf.keras.Model(inputs=[visual_input, proprioceptive_input, somatosensory_input, goal_input],
                                  outputs=[policy_outputs, value_outputs])

    return policy, value, policy_value


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sequence_length = 10

    policy, value, policy_value = build_shadow_brain(gym.make("ShadowHand-v0"))
    tf.keras.utils.plot_model(policy, show_shapes=True, expand_nested=True)
    input_data = [(tf.random.normal([16, sequence_length, 200, 200, 3]),
                   tf.random.normal([16, sequence_length, 24]),
                   tf.random.normal([16, sequence_length, 32]),
                   tf.random.normal([16, sequence_length, 4]))]

    start_time = time.time()
    output = policy(input_data)
    print(f"Execution Time: {time.time() - start_time}")

    print(tf.reduce_all(policy.weights[0] == value.weights[0]))
    policy.weights[0] = policy.weights[0] * 2
    print(tf.reduce_all(policy.weights[0] == value.weights[0]))
