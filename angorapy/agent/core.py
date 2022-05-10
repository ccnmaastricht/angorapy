#!/usr/bin/env python
"""Core methods providing functionality to the agent."""
from itertools import accumulate
from typing import List

import numpy as np
import tensorflow as tf
from scipy.signal import lfilter

from angorapy.common.const import NP_FLOAT_PREC


def get_discounted_returns(reward_trajectory, discount_factor: float):
    """Discounted future rewards calculation using itertools. Way faster than list comprehension."""
    return list(accumulate(reward_trajectory[::-1], lambda previous, x: previous * discount_factor + x))[::-1]


def estimate_advantage(rewards: List, values: List, t_is_terminal: List, gamma: float, lam: float) -> np.ndarray:
    """K-Step return estimator for Generalized Advantage Estimation.
    From: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. (Schulman et. al., 2018)

    Using the raw discounted future reward suffers from high variance and hence could hinder learning. Using a k-step
    return estimator introduces some bias but reduces variance, yielding a beneficial trade-off.

    :param rewards:             list of rewards where r[t] is the reward from taking a[t] in state serialization[t],
                                transitioning to serialization[t + 1]
    :param values:              value estimations for state trajectory. Requires a value for last state not covered in
                                rewards, too, as this might be non-terminal
    :param t_is_terminal        boolean indicator list of false for non-terminal and true for terminal states. A
                                terminal state is one after which the absorbing state follows and the episode ends.
                                Hence if t is terminal, this is not the last state observed but the last in which
                                experience can be collected through taking an step_tuple.
    :param gamma:               a discount factor weighting the importance of future rewards
    :param lam:                 GAE'serialization lambda parameter compromising between bias and variance. High lambda
                                results in
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


def estimate_episode_advantages(rewards, values, discount, lam):
    """Estimate advantage of a single episode (or part of it).

    Taken from Open AI'serialization spinning up repository."""
    deltas = np.array(rewards) + discount * np.array(values[1:]) - np.array(values[:-1])
    return lfilter([1], [1, float(-(discount * lam))], deltas[::-1], axis=0)[::-1].astype(NP_FLOAT_PREC)


@tf.function
def extract_discrete_action_probabilities(predictions: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    """Given a tensor of predictions with shape [batch_size, sequence, n_actions] or [batch_size, n_actions] and a 2D or
    1D tensor of actions with shape [batch_size, sequence_length] or [batch_size] extract the probabilities for the
    actions."""
    assert len(predictions.shape) in [2, 3], "Predictions should be a tensor of rank 2 or 3."
    assert len(actions.shape) in [1, 2], "Actions should be a tensor of rank 1 or 2."

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
