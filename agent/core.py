import random
from abc import ABC, abstractmethod

import numpy
import tensorflow as tf

from itertools import accumulate

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
        self.state_dimensionality = state_dimensionality

    def act(self, state: numpy.ndarray):
        return random.randrange(self.n_actions)

    def drill(self, **kwargs):
        pass


def get_discounted_returns(reward_trajectory, discount_factor: float):
    """Discounted future rewards calculation using itertools. Way faster than list comprehension."""
    return list(accumulate(reward_trajectory[::-1], lambda previous, x: previous * discount_factor + x))[::-1]
