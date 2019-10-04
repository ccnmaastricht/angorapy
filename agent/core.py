import random
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import List

import numpy
import scipy.signal

import tensorflow as tf
from utilities.util import env_extract_dims


class RLAgent(ABC):

    def __init__(self):
        self.iteration = 0

    # @abstractmethod
    # def act(self, state: numpy.ndarray):
    #     """Agent determines a favourable action following his policy, given a state."""
    #     pass

    @abstractmethod
    def drill(self, **kwargs):
        """Train the agent on a given environment."""
        pass


class RandomAgent(RLAgent):

    def __init__(self, env):
        super().__init__()
        self.state_dimensionality, self.n_actions = env_extract_dims(env)

    def act(self, state: numpy.ndarray):
        return tf.convert_to_tensor(random.randrange(self.n_actions)), 1/self.n_actions

    def drill(self, **kwargs):
        pass


def get_discounted_returns(reward_trajectory, discount_factor: float):
    """Discounted future rewards calculation using itertools. Way faster than list comprehension."""
    return list(accumulate(reward_trajectory[::-1], lambda previous, x: previous * discount_factor + x))[::-1]


def estimate_advantage(rewards: numpy.ndarray, values: numpy.ndarray,
                       t_is_terminal: List, gamma: float, gae_lambda: float) -> numpy.ndarray:
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
    :param gae_lambda:          GAE's lambda parameter compromising between bias and variance. High lambda results in
                                less bias but more variance. 0 < Lambda < 1

    :return:                    the estimations about the returns of a trajectory
    """
    if numpy.size(rewards, 0) - numpy.size(values, 0) != -1:
        raise ValueError(
            "For horizoned GAE the values need also to include a prediction for the very last state observed, however "
            "the given values list is not one element longer than the given rewards list.")

    total_steps = numpy.size(rewards, 0)
    return_estimations = numpy.ndarray(shape=(total_steps,))

    previous = 0
    for t in reversed(range(total_steps)):
        if t_is_terminal[t]:
            previous = 0

        if t_is_terminal[t]:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + (gamma * values[t + 1]) - values[t]
        previous = delta + gamma * gae_lambda * previous

        return_estimations[t] = previous

    return return_estimations


def pdf_normal(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + numpy.log(2 * numpy.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def entropy_normal(x, mu, log_std):
    pass



if __name__ == "__main__":
    one_episode = numpy.array(list(range(1, 10)))
    ep_values = numpy.array(get_discounted_returns(one_episode, 0.99))

    ep_dones = ([False] * (len(one_episode) - 1) + [True])

    rewards = one_episode
    values = numpy.concatenate((ep_values, [0]))
    dones = ep_dones

    rewards = rewards[:-2]
    values = values[:-2]
    dones = dones[:-2]

    advs = estimate_advantage(rewards,
                              values,
                              dones,
                              gamma=0.99,
                              gae_lambda=0.95)

    # values = values[:-1]
    # deltas = rewards[:-1] + 0.99 * values[1:] - values[:-1]
    # oai_advs = discount_cumsum(deltas, 0.99 * 0.95)

    print(advs)