#!/usr/bin/env python
"""Methods for sampling an action given a policy and a state for discrete and continuous actions spaces."""
import os
from typing import Tuple, Union

import numpy as np
import tensorflow as tf

from agent.core import gaussian_log_pdf


def act_discrete(log_probabilities: Union[tf.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Sample an action from a discrete action distribution predicted by the given policy for a given state."""
    assert isinstance(log_probabilities, tf.Tensor) or isinstance(log_probabilities, np.ndarray), \
        f"Policy methods (act_discrete) require Tensors or Numpy Arrays as input, " \
        f"not {type(log_probabilities).__name__}."

    if tf.rank(log_probabilities) == 3:
        # there appears to be a sequence dimension
        assert log_probabilities.shape[1] == 1, "Policy actions can only be selected for a single timestep, but the " \
                                                "dimensionality of the given tensor is more than 1 at rank 1."

        log_probabilities = tf.squeeze(log_probabilities, axis=1)

    action = tf.random.categorical(log_probabilities, 1)[0][0]

    return action.numpy(), log_probabilities[0][action]


def act_continuous(means: tf.Tensor, log_stdevs: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Sample an action from a continuous action distribution predicted by the given policy for a given state."""
    actions = tf.random.normal(means.shape, means, tf.exp(log_stdevs))
    probabilities = gaussian_log_pdf(actions, means=means, log_stdevs=log_stdevs)

    return tf.reshape(actions, [-1]).numpy(), tf.squeeze(probabilities).numpy()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    out = act_discrete(tf.sigmoid(tf.random.normal([16, 4])))
    print(out)
