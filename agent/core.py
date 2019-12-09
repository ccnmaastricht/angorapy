#!/usr/bin/env python
"""Core methods providing functionality to the agent."""
import math
from itertools import accumulate
from pprint import pprint
from typing import List

import numpy as np
import tensorflow as tf
from scipy.signal import lfilter


# RETURN/ADVANTAGE CALCULATION

def get_discounted_returns(reward_trajectory, discount_factor: float):
    """Discounted future rewards calculation using itertools. Way faster than list comprehension."""
    return list(accumulate(reward_trajectory[::-1], lambda previous, x: previous * discount_factor + x))[::-1]


def estimate_advantage(rewards: List, values: List, t_is_terminal: List, gamma: float, lam: float) -> np.ndarray:
    """K-Step return estimator for Generalized Advantage Estimation.
    From: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. (Schulman et. al., 2018)

    Using the raw discounted future reward suffers from high variance and hence could hinder learning. Using a k-step
    return estimator introduces some bias but reduces variance, yielding a beneficial trade-off.

    :param rewards:             list of rewards where r[t] is the reward from taking a[t] in state s[t], transitioning
                                to s[t + 1]
    :param values:              value estimations for state trajectory. Requires a value for last state not covered in
                                rewards, too, as this might be non-terminal
    :param t_is_terminal        boolean indicator list of false for non-terminal and true for terminal states. A
                                terminal state is one after which the absorbing state follows and the episode ends.
                                Hence if t is terminal, this is not the last state observed but the last in which
                                experience can be collected through taking an action.
    :param gamma:               a discount factor weighting the importance of future rewards
    :param lam:                 GAE's lambda parameter compromising between bias and variance. High lambda results in
                                less bias but more variance. 0 < Lambda < 1

    :return:                    the estimations about the returns of a trajectory
    """
    if np.size(rewards, 0) - np.size(values, 0) != -1:
        raise ValueError("Values must include one more prediction than there are rewards.")

    total_steps = np.size(rewards, 0)
    return_estimations = np.ndarray(shape=(total_steps,)).astype(np.float32)

    previous = 0
    for t in reversed(range(total_steps)):
        if t_is_terminal[t]:
            previous = 0

        delta = rewards[t] - values[t]
        if not t_is_terminal[t]:
            delta += (gamma * values[t + 1])
        previous = delta + gamma * lam * previous

        return_estimations[t] = previous

    return return_estimations


def estimate_episode_advantages(rewards, values, gamma, lam):
    """Estimate advantage of a single episode (or part of it), taken from Open AI's spinning up repository."""
    deltas = np.array(rewards, dtype=np.float32) + gamma * np.array(values[1:], dtype=np.float32) - np.array(
        values[:-1], dtype=np.float32)
    return lfilter([1], [1, float(-(gamma * lam))], deltas[::-1], axis=0)[::-1].astype(np.float32)


# PROBABILITY


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

# MANIPULATION

@tf.function
def extract_discrete_action_probabilities(predictions: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    """Given a tensor of predictions with shape [batch_size, sequence, n_actions] or [batch_size, n_actions] and a 2D or
    1D tensor of actions with shape [batch_size, sequence_length] or [batch_size] extract the probabilities for the
    actions."""
    assert len(actions.shape) in [1, 2], "Actions should be a tensor of rank 1 or 2."
    assert len(predictions.shape) in [2, 3], "Predictions should be a tensor of rank 2 or 3."

    if len(actions.shape) == 1:
        indices = tf.concat([tf.reshape(tf.range(actions.shape[0]), [-1, 1]), tf.reshape(actions, [-1, 1])], axis=-1)
        choices = tf.gather_nd(predictions, indices)
    else:
        batch_indices = tf.reshape(
            tf.tile(tf.expand_dims(tf.range(actions.shape[0]), axis=-1), [1, actions.shape[1]]), [-1, 1])
        sequence_indices = tf.reshape(
            tf.tile(tf.expand_dims(tf.range(actions.shape[1]), axis=0), [1, actions.shape[0]]), [-1, 1])

        indices = tf.concat((batch_indices, sequence_indices, tf.reshape(actions, [-1, 1])), axis=-1)

        choices = tf.gather_nd(predictions, indices)
        choices = tf.reshape(choices, actions.shape)

    return choices


if __name__ == "__main__":
    from scipy.stats import norm
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.config.experimental_run_functions_eagerly(True)

    x = tf.convert_to_tensor([[5, 1], [2, 3]], dtype=tf.float32)
    mu = tf.convert_to_tensor([[2, 3], [2, 1]], dtype=tf.float32)
    sig = tf.convert_to_tensor([[1, 1], [4, 1]], dtype=tf.float32)

    out_np = np.sum(norm.logpdf(x, loc=mu, scale=sig), axis=-1)
    out_logged = tf.math.log(gaussian_pdf(x, mu, sig)).numpy()
    out_log_pdf = gaussian_log_pdf(x, mu, tf.math.log(sig)).numpy()

    pprint((out_np, out_logged, out_log_pdf))
    print(out_logged - out_log_pdf)
