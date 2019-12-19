"""Probability functions."""

import math

import numpy as np
import tensorflow as tf


@tf.function
def gaussian_pdf(samples: tf.Tensor, means: tf.Tensor, stdevs: tf.Tensor):
    """Calculate probability density for a given batch of potentially joint Gaussian PDF."""
    samples_transformed = (samples - means) / stdevs
    pdf = (tf.exp(-(tf.pow(samples_transformed, 2) / 2)) / tf.sqrt(2 * math.pi)) / stdevs
    return tf.math.reduce_prod(pdf, axis=-1)


@tf.function
def gaussian_log_pdf(samples: tf.Tensor, means: tf.Tensor, log_stdevs: tf.Tensor):
    """Calculate log probability density for a given batch of potentially joint Gaussian PDF.

    Input Shapes:
        - all: (B, A) or (B, S, A)

    Output Shapes:
        - all: (B) or (B, S)
    """
    log_likelihoods = (- tf.reduce_sum(log_stdevs, axis=-1)
                       - tf.math.log(2 * np.pi)
                       - (0.5 * tf.reduce_sum(tf.square(((samples - means) / tf.exp(log_stdevs))), axis=-1)))

    # log_likelihoods = tf.math.log(gaussian_pdf(samples, means, tf.exp(log_stdevs)))

    return log_likelihoods


@tf.function
def gaussian_entropy(stdevs: tf.Tensor):
    """Calculate the joint entropy of Gaussian random variables described by their log standard deviations.

    Input Shape: (B, A) or (B, S, A) for recurrent

    Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
    of the joint entropy and the sum of marginal entropies.
    """
    entropy = .5 * tf.math.log(2 * math.pi * math.e * tf.pow(stdevs, 2))
    return tf.reduce_sum(entropy, axis=-1)


@tf.function
def gaussian_entropy_from_log(log_stdevs: tf.Tensor):
    """Calculate the joint entropy of Gaussian random variables described by their log standard deviations.

    Input Shape: (B, A) or (B, S, A) for recurrent

    Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
    of the joint entropy and the sum of marginal entropies.
    """
    entropy = .5 * (tf.math.log(math.pi * 2) + (tf.multiply(2.0, log_stdevs) + 1.0))
    return tf.reduce_sum(entropy, axis=-1)


@tf.function
def categorical_entropy(pmf: tf.Tensor):
    """Calculate entropy of a categorical distribution, where the pmf is given as log probabilities."""
    return - tf.reduce_sum(tf.math.log(pmf) * pmf, axis=-1)


@tf.function
def categorical_entropy_from_log(pmf: tf.Tensor):
    """Calculate entropy of a categorical distribution, where the pmf is given as log probabilities."""
    return - tf.reduce_sum(tf.exp(pmf) * pmf, axis=-1)


@tf.function
def approximate_kl_divergence(log_pa, log_pb):
    """Approximate KL-divergence between distributions a and b where some sample has probability pa in a and pb in b."""
    return .5 * tf.reduce_mean(tf.square(log_pa - log_pb))