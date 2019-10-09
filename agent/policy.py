#!/usr/bin/env python
"""Policy Wrappers."""
from typing import Tuple

import numpy
import tensorflow as tf

from agent.core import gaussian_pdf


def act_discrete(policy, state: tf.Tensor) -> Tuple[numpy.ndarray, numpy.ndarray]:
    probabilities = policy(state, training=False)
    action = tf.random.categorical(tf.math.log(probabilities), 1)[0][0]

    return action.numpy(), probabilities[0][action].numpy()


def act_continuous(policy, state: tf.Tensor) -> Tuple[numpy.ndarray, numpy.ndarray]:
    multivariates = policy(state, training=False)
    n_actions = multivariates.shape[1] // 2
    means = multivariates[:, :n_actions]
    stdevs = multivariates[:, n_actions:]

    actions = tf.random.normal([state.shape[0], n_actions], means, stdevs, dtype=tf.float64)
    probabilities = gaussian_pdf(actions, means=means, stdevs=stdevs)

    return tf.reshape(actions, [-1]).numpy(), tf.squeeze(probabilities).numpy()
