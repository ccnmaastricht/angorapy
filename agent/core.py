import math
from itertools import accumulate
from typing import List

import numpy
import tensorflow as tf


def get_discounted_returns(reward_trajectory, discount_factor: float):
    """Discounted future rewards calculation using itertools. Way faster than list comprehension."""
    return list(accumulate(reward_trajectory[::-1], lambda previous, x: previous * discount_factor + x))[::-1]


def estimate_advantage(rewards: List, values: List, t_is_terminal: List, gamma: float, lam: float) -> numpy.ndarray:
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
    :param lam:          GAE's lambda parameter compromising between bias and variance. High lambda results in
                                less bias but more variance. 0 < Lambda < 1

    :return:                    the estimations about the returns of a trajectory
    """
    if numpy.size(rewards, 0) - numpy.size(values, 0) != -1:
        raise ValueError("Values must include one more prediction than there are states.")

    total_steps = numpy.size(rewards, 0)
    return_estimations = numpy.ndarray(shape=(total_steps,)).astype(numpy.float32)

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


def normalize_advantages(advantages: numpy.ndarray) -> numpy.ndarray:
    """Z-score standardization of advantages if activated."""
    return (advantages - advantages.mean()) / advantages.std()


def gaussian_pdf(samples: tf.Tensor, means: tf.Tensor, stdevs: tf.Tensor):
    samples_transformed = (samples - means) / stdevs
    pdf = (tf.exp(-(tf.pow(samples_transformed, 2) / 2)) / tf.sqrt(2 * math.pi)) / stdevs
    return tf.reduce_sum(pdf, axis=1)


def gaussian_entropy(stdevs: tf.Tensor):
    entropy = .5 * tf.math.log(2 * math.pi * math.e * tf.pow(stdevs, 2))
    return tf.reduce_sum(entropy, axis=1)


def categorical_entropy(pmf: tf.Tensor):
    return - tf.reduce_sum(pmf * tf.math.log(pmf), 1)


if __name__ == "__main__":
    gaussian_pdf(tf.convert_to_tensor([[2, 3]]),
                 tf.convert_to_tensor([[2, 3]]),
                 tf.convert_to_tensor([[1, 1]]))

    gaussian_entropy(tf.convert_to_tensor([[1, 2]]))
