#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time
from typing import Iterable

import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD

from environments import *
from models.components import _build_visual_encoder, _build_non_visual_component, _build_continuous_head, \
    _build_discrete_head
from utilities.util import env_extract_dims


def build_shadow_brain(env: gym.Env, batch_size: int):
    """Build network for the shadow hand task."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32

    # inputs
    visual_in = tf.keras.Input(batch_shape=(batch_size, None, 200, 200, 3), name="visual_input")
    proprio_in = tf.keras.Input(batch_shape=(batch_size, None, 48,), name="proprioceptive_input")
    touch_in = tf.keras.Input(batch_shape=(batch_size, None, 92,), name="somatosensory_input")
    goal_in = tf.keras.Input(batch_shape=(batch_size, None, 7,), name="goal_input")

    # abstractions of perceptive inputs
    visual_latent = TD(_build_visual_encoder(shape=(200, 200, 3), batch_size=batch_size))(visual_in)
    proprio_latent = TD(_build_non_visual_component(48, 12, 8, batch_size=batch_size))(proprio_in)
    touch_latent = TD(_build_non_visual_component(92, 24, 8, batch_size=batch_size))(touch_in)

    # concatenation of perceptive abstractions
    concatenation = tf.keras.layers.Concatenate()([visual_latent, proprio_latent, touch_latent])

    # fully connected ReLu block integrating perceptive representations
    x = TD(tf.keras.layers.Dense(48))(concatenation)
    x = TD(tf.keras.layers.ReLU())(x)
    x = TD(tf.keras.layers.Dense(32))(x)
    x = TD(tf.keras.layers.ReLU())(x)
    x.set_shape([batch_size] + x.shape[1:])
    x = tf.keras.layers.Concatenate()([x, goal_in])

    # recurrent layer
    o = tf.keras.layers.LSTM(hidden_dimensions, stateful=True, batch_size=batch_size)(x)

    # output heads
    policy_out = _build_continuous_head(n_actions, o) if continuous_control else _build_discrete_head(n_actions, o)
    value_out = tf.keras.layers.Dense(1)(o)

    # define separate and joint models
    policy = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in], outputs=[policy_out])
    value = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in], outputs=[value_out])
    joint = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in], outputs=[policy_out, value_out])

    return policy, value, joint


def init_hidden(shape: Iterable):
    """Get initial hidden state"""
    return tf.zeros(shape=shape)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sequence_length = 10
    batch_size = 1

    pi, vn, pv = build_shadow_brain(gym.make("ShadowHand-v1"), batch_size=1)
    tf.keras.utils.plot_model(pi, show_shapes=True, expand_nested=True)
    pv.summary()

    optimizer = tf.keras.optimizers.Adam()
    start_time = time.time()
    with tf.device("GPU:0"):
        for t in range(sequence_length):
            input_data = (tf.random.normal([batch_size, 1, 200, 200, 3]).numpy(),
                          tf.random.normal([batch_size, 1, 48]).numpy(),
                          tf.random.normal([batch_size, 1, 92]).numpy(),
                          tf.random.normal([batch_size, 1, 7]).numpy())
            with tf.GradientTape() as tape:
                out, v = pv(input_data, training=True)
                loss = tf.math.reduce_sum(out * v)

            grads = tape.gradient(loss, pv.trainable_variables)
            optimizer.apply_gradients(zip(grads, pv.trainable_variables))

    print(f"Execution Time: {time.time() - start_time}")
