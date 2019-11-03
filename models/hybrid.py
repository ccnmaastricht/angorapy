#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time

import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as TimeDist

from models.components import build_visual_component, build_non_visual_component


def build_shadow_brain(action_space):
    visual_input = tf.keras.Input(shape=(None, 200, 200, 3), name="visual_input")
    proprioceptive_input = tf.keras.Input(shape=(None, 24,), name="proprioceptive_input")
    somatosensory_input = tf.keras.Input(shape=(None, 32,), name="somatosensory_input")
    goal_input = tf.keras.Input(shape=(None, 4,), name="goal_input")

    visual_abstraction = TimeDist(build_visual_component())(visual_input)
    proprioceptive_abstraction = TimeDist(build_non_visual_component(24, 12, 8))(
        proprioceptive_input)
    somatosensory_abstraction = TimeDist(build_non_visual_component(32, 24, 8))(
        somatosensory_input)

    concatenation = tf.keras.layers.Concatenate()([visual_abstraction,
                                                   proprioceptive_abstraction,
                                                   somatosensory_abstraction])

    x = TimeDist(tf.keras.layers.Dense(48))(concatenation)
    x = TimeDist(tf.keras.layers.Dense(32))(x)
    x = tf.keras.layers.Concatenate()([x, goal_input])

    x = tf.keras.layers.LSTM(32, )(x)
    outputs = tf.keras.layers.Dense(action_space)(x)

    return tf.keras.Model(inputs=[visual_input, proprioceptive_input, somatosensory_input, goal_input],
                          outputs=outputs)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sequence_length = 10

    network = build_shadow_brain(3)
    tf.keras.utils.plot_model(network, show_shapes=True, expand_nested=True)
    input_data = [(tf.random.normal([16, sequence_length, 200, 200, 3]),
                   tf.random.normal([16, sequence_length, 24]),
                   tf.random.normal([16, sequence_length, 32]),
                   tf.random.normal([16, sequence_length, 4]))]

    start_time = time.time()
    output = network(input_data)
    print(f"Execution Time: {time.time() - start_time}")
