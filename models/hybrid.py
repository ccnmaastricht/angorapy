#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time
from typing import Iterable

import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow_core.python.keras.utils import plot_model
from tqdm import tqdm

from environments import *
from models.components import _build_fcn_component, _build_continuous_head, _build_discrete_head
from models.convolutional import _build_visual_encoder
from utilities.util import env_extract_dims


def build_shadow_brain(env: gym.Env, bs: int):
    """Build network for the shadow hand task."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32

    # inputs
    visual_in = tf.keras.Input(batch_shape=(bs, None, 224, 224, 3), name="visual_input")
    proprio_in = tf.keras.Input(batch_shape=(bs, None, 48,), name="proprioceptive_input")
    touch_in = tf.keras.Input(batch_shape=(bs, None, 92,), name="somatosensory_input")
    goal_in = tf.keras.Input(batch_shape=(bs, None, 7,), name="goal_input")

    # abstractions of perceptive inputs
    visual_latent = TD(_build_visual_encoder(shape=(224, 224, 3), batch_size=bs, name="latent_vision"))(visual_in)
    proprio_latent = TD(_build_fcn_component(48, 12, 8, batch_size=bs, name="latent_proprio"))(proprio_in)
    touch_latent = TD(_build_fcn_component(92, 24, 8, batch_size=bs, name="latent_touch"))(touch_in)

    # concatenation of perceptive abstractions
    concatenation = tf.keras.layers.Concatenate()([visual_latent, proprio_latent, touch_latent])

    # fully connected layer integrating perceptive representations
    x = TD(tf.keras.layers.Dense(48))(concatenation)
    x = TD(tf.keras.layers.ReLU())(x)

    # concatenation of goal and perception
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.Concatenate()([x, goal_in])

    # recurrent layer
    o = tf.keras.layers.GRU(hidden_dimensions, stateful=True, return_sequences=True, batch_size=bs)(x)

    # output heads
    policy_out = _build_continuous_head(n_actions, o) if continuous_control else _build_discrete_head(n_actions, o)
    value_out = tf.keras.layers.Dense(1, name="value")(o)

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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    sequence_length = 100
    batch_size = 256

    env = gym.make("ShadowHand-v1")
    _, _, joint = build_shadow_brain(env, bs=batch_size)
    plot_model(joint, to_file="model.png")
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD()

    @tf.function
    def _train():
        start_time = time.time()

        for _ in tqdm(range(sequence_length), disable=False):
            sample_batch = (tf.convert_to_tensor(tf.random.normal([batch_size, 4, 224, 224, 3])),
                            tf.convert_to_tensor(tf.random.normal([batch_size, 4, 48])),
                            tf.convert_to_tensor(tf.random.normal([batch_size, 4, 92])),
                            tf.convert_to_tensor(tf.random.normal([batch_size, 4, 7])))

            with tf.GradientTape() as tape:
                out, v = joint(sample_batch, training=True)
                loss = tf.reduce_mean(out - v)

            grads = tape.gradient(loss, joint.trainable_variables)
            optimizer.apply_gradients(zip(grads, joint.trainable_variables))

        print(f"Execution Time: {time.time() - start_time}")

    _train()
