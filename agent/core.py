import random
from abc import ABC, abstractmethod
from itertools import accumulate

import numpy


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


def generalized_advantage_estimator(rewards: numpy.ndarray, values: numpy.ndarray, gamma: float, gae_lambda: float):
    """K-Step return Estimator for Generalized Advantage Estimation.
    From: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. (Schulman et. al., 2018)

    Using the raw discounted future reward suffers from high variance and hence could hinder learning. Using a k-step
    return estimator introduces some bias but reduces variance, yielding a beneficial trade-off."""
    total_steps = len(rewards)
    return_estimations = numpy.ndarray((total_steps,))

    previous = 0
    for t in reversed(range(total_steps)):
        # -2 because for terminal state can't value next state and for pre-terminal state there is no point in judging
        # the value of terminal state as it will finish here anyways and no action influences that
        if t < total_steps - 2:
            delta = rewards[t] + (gamma * values[t + 1]) - values[t]
        else:
            delta = rewards[t] - values[t]
        previous = delta + gamma * gae_lambda * previous
        return_estimations[t] = previous

    return return_estimations


if __name__ == "__main__":
    rewards = numpy.array(list(range(1, 10)))
    values = numpy.array(get_discounted_returns(rewards, 0.99))
    advs = generalized_advantage_estimator(
        rewards,
        values,
        gamma=0.99,
        gae_lambda=0.95
    )

    print(rewards)
    print(values)
    print(advs)
    print(advs + values)
