#!/usr/bin/env python
"""Normalization methods."""

import tensorflow as tf
from tensorflow_core import Tensor


class RunningNormalization(tf.keras.layers.Layer):
    """Normalization Layer that apply z-score normalization for a given input batch.

    Z-score transformation normalizes the input to be of 0 mean and standard deviation 1 by subtracting the mean of
    the overall data and dividing by the standard deviation. Since the data distribution is not known, the layer
    keeps track of running means and standard deviations, based on which the transformation is applied.
    """
    running_means: Tensor
    running_stdevs: Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.total_samples_observed = 0

    def build(self, input_shape):
        self.running_means = tf.zeros(input_shape[-1])
        self.running_stdevs = tf.ones(input_shape[-1])

    def call(self, batch: tf.Tensor, **kwargs) -> tf.Tensor:
        """Normalize a given batch of 1D tensors and update running mean and std."""

        return batch

        # calculate statistics for the batch
        batch_len = batch.shape[0] if batch.shape[0] is not None else 0
        batch_means = tf.reduce_mean(batch, axis=0)
        batch_stdevs = tf.math.reduce_std(batch, axis=0)

        if batch_len > 0:  # protect against building
            # calculate weights based on number of seen examples
            weight_experience = self.total_samples_observed / (self.total_samples_observed + batch_len)
            weight_batch = 1 - weight_experience

            # update statistics
            self.running_means = tf.multiply(weight_experience, self.running_means) + tf.multiply(weight_batch,
                                                                                                  batch_means)
            self.running_stdevs = weight_experience * self.running_stdevs + weight_batch * batch_stdevs

        # normalize
        return (batch - self.running_means) / (self.running_stdevs + 1e-8)
