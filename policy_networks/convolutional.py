#!/usr/bin/env python
"""Convolutional Networks serving as policy or critic for agents getting visual input."""

import tensorflow as tf


class PPOActorCNN(tf.keras.Model):
    """Fully-connected network taking the role of an actor."""

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

    def call(self, input_tensor, training=False, **kwargs):
        return 0


class PPOCriticCNN(tf.keras.Model):
    """Fully-connected network taking the role of an actor."""

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

    def call(self, input_tensor, training=False, **kwargs):
        return 0
