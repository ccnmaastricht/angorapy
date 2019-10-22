#!/usr/bin/env python
"""Convolutional Networks serving as policy or critic for agents getting visual input."""
import numpy
import tensorflow as tf
from gym.spaces import Box

from utilities.util import env_extract_dims


class PPOActorCNN(tf.keras.Model):
    """Fully-connected network taking the role of an actor."""

    def __init__(self, env):
        super().__init__()

        self.continuous_control = isinstance(env.action_space, Box)
        self.state_dimensionality, self.n_actions = env_extract_dims(env)

        self.convolver = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 8, 4, input_shape=(30, 30, 1), activation="relu"),
            tf.keras.layers.Conv2D(64, 3, 2, activation="relu"),
            tf.keras.layers.ZeroPadding2D(1, data_format="channels_last"),
            tf.keras.layers.Conv2D(128, 3, 1, activation="relu"),
        ])

        self.forward = tf.keras.Sequential()
        self.forward.add(tf.keras.layers.Flatten())
        self.forward.add(tf.keras.layers.Dense(128, activation="relu"))
        self.forward.add(tf.keras.layers.Dense(64, activation="relu"))
        self.forward.add(tf.keras.layers.Dense(32, activation="relu"))
        self.forward.add(tf.keras.layers.Dense(self.n_actions, activation="softmax"))

        example_input = numpy.expand_dims(env.reset().astype(numpy.float32), axis=0)
        self.predict(example_input)

    def call(self, input_tensor, training=False, **kwargs):
        convolved = self.convolver(input_tensor)
        out = self.forward(convolved)

        return out


class PPOCriticCNN(tf.keras.Model):
    """Fully-connected network taking the role of an actor."""

    def __init__(self, env):
        super().__init__()

        self.continuous_control = isinstance(env.action_space, Box)
        self.state_dimensionality, self.n_actions = env_extract_dims(env)

        self.convolver = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 8, 4, input_shape=(30, 30, 1), activation="relu"),
            tf.keras.layers.Conv2D(64, 3, 2, activation="relu"),
            tf.keras.layers.ZeroPadding2D(1, data_format="channels_last"),
            tf.keras.layers.Conv2D(128, 3, 1, activation="relu"),
        ])

        self.forward = tf.keras.Sequential()
        self.forward.add(tf.keras.layers.Flatten())
        self.forward.add(tf.keras.layers.Dense(128, activation="relu"))
        self.forward.add(tf.keras.layers.Dense(64, activation="relu"))
        self.forward.add(tf.keras.layers.Dense(32, activation="relu"))
        self.forward.add(tf.keras.layers.Dense(1, activation="linear"))

        example_input = numpy.expand_dims(env.reset().astype(numpy.float32), axis=0)
        self.predict(example_input)

    def call(self, input_tensor, training=False, **kwargs):
        convolved = self.convolver(input_tensor)
        out = self.forward(convolved)

        return out
