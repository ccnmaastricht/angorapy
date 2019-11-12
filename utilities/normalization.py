#!/usr/bin/env python
"""Normalization methods."""
import os

import tensorflow as tf
from tensorflow_core import Tensor


class RunningNormalization(tf.keras.layers.Layer):
    """Normalization Layer that apply z-score normalization for a given input batch.

    Z-score transformation normalizes the input to be of 0 mean and standard deviation 1 by subtracting the mean of
    the overall data and dividing by the standard deviation. Since the data distribution is not known, the layer
    keeps track of running means and standard deviations, based on which the transformation is applied.
    """
    mu: Tensor
    std: Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n = 0

    def build(self, input_shape):
        self.mu = tf.zeros(input_shape[-1])
        self.std = tf.ones(input_shape[-1])

    def call(self, batch: tf.Tensor, **kwargs) -> tf.Tensor:
        """Normalize a given batch of 1D tensors and update running mean and std."""

        # calculate statistics for the batch
        batch_len = batch.shape[0] if batch.shape[0] is not None else 0
        batch_mu = tf.reduce_mean(batch, axis=0)
        batch_std = tf.math.reduce_std(batch, axis=0)

        mu_old = tf.convert_to_tensor(self.mu.numpy())

        if batch_len > 0:  # protect against building
            # calculate weights based on number of seen examples
            weight_experience = self.n / (self.n + batch_len)
            weight_batch = 1 - weight_experience

            # update statistics
            self.mu = tf.multiply(weight_experience, self.mu) + tf.multiply(weight_batch, batch_mu)
            self.std = tf.sqrt((self.n * self.std ** 2 + batch_len * batch_std ** 2 + self.n * (
                        mu_old - self.mu) ** 2 + batch_len * (batch_mu - self.mu) ** 2) / (batch_len + self.n))

        self.n += batch_len

        # normalize
        return (batch - self.mu) / (self.std + 1e-8)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    normalizer = RunningNormalization()

    std = tf.convert_to_tensor(list(range(1, 7)), dtype=tf.float32)
    mean = tf.convert_to_tensor(list(range(1, 7)), dtype=tf.float32)
    inputs = [tf.random.normal([5, 6]) for _ in range(1000)]
    all = tf.concat(inputs, axis=0)

    print(tf.reduce_mean(all))
    print(tf.math.reduce_std(all))

    for batch in inputs:
        normalizer(batch * std + mean)

    print(normalizer.mu)
    print(normalizer.std)
