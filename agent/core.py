import random
from abc import ABC, abstractmethod

import numpy
import tensorflow as tf


class _RLAgent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def act(self, state: numpy.ndarray):
        """Agent determines a favourable action following his policy, given a state."""
        pass

    @abstractmethod
    def drill(self, **kwargs):
        """Train the agent on a given environment."""
        pass


class RandomAgent(_RLAgent):

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.state_dimensionality = state_dtimensionality

    def act(self, state: numpy.ndarray):
        return random.randrange(self.n_actions)

    def drill(self, **kwargs):
        pass


def get_discounted_returns(reward_trajectory, discount_factor: tf.Tensor):
    return [numpy.sum([discount_factor**k * r for k, r in enumerate(reward_trajectory[t:])]) for t in range(len(reward_trajectory))]