#!/usr/bin/env python
"""Methods for sampling an action given a policy and a state for discrete and continuous actions spaces."""
import os
from typing import Tuple

import numpy
import tensorflow as tf

from agent.core import gaussian_pdf


def act_discrete(probabilities: tf.Tensor) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Sample an action from a discrete action distribution predicted by the given policy for a given state."""
    action = tf.random.categorical(tf.math.log(probabilities), 1)[0][0]

    return action.numpy(), probabilities[0][action]


def act_continuous(multivariates: tf.Tensor) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Sample an action from a continuous action distribution predicted by the given policy for a given state."""
    means, stdevs = tf.split(multivariates, 2, axis=-1)

    actions = tf.random.normal(means.shape, means, stdevs)
    probabilities = gaussian_pdf(actions, means=means, stdevs=stdevs)

    return tf.reshape(actions, [-1]).numpy(), tf.squeeze(probabilities).numpy()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    out = act_discrete(tf.sigmoid(tf.random.normal([16, 4])))
    print(out)
