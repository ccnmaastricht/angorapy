#!/usr/bin/env python
"""Custom layer components to be used in model implementations."""
import tensorflow as tf


class StdevLayer(tf.keras.layers.Layer):
    """Layer carrying parameters to be directly optimized, serving as standard deviations in continuous gaussian
    policies."""

    def __init__(self, n_actions, **kwargs):
        super(StdevLayer, self).__init__(**kwargs)
        self.n_actions = n_actions

    def build(self, input_shape):
        """Build the layer by adding the parameters representing the standard deviations."""

        # initializing the log stdevs as zeros essentially means we initialize the true standard deviations to 1
        self.log_stdevs = self.add_weight("log_stdevs", shape=[self.n_actions, 1], initializer='zeros', trainable=True)

    def call(self, input, **kwargs):
        return tf.matmul(tf.ones_like(input), self.log_stdevs)


class BetaDistributionSpreadLayer(tf.keras.layers.Layer):
    """Layer carrying parameters to be directly optimized, serving as v = a + b values in reparameterized beta policies.
    """

    def __init__(self, n_actions, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = n_actions

    def build(self, input_shape):
        """Build the layer by adding the parameters representing the standard deviations."""

        # initializing the log stdevs as zeros essentially means we initialize to 2
        self.spreads = self.add_weight("sum_ab", shape=[self.n_actions, 1], initializer='zeros', trainable=True)

    def call(self, input, **kwargs):
        return tf.matmul(tf.ones_like(input), self.spreads)
