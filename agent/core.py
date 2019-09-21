import random
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import List

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


def generalized_advantage_estimator(reward_trajectory: List[int], value_predictions: List[int], discount_factor: float, gae_lambda: float, k):
    """K-Step return Estimator for Generalized Advantage Estimation.
    From: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. (Schulman et. al., 2018)

    Using the raw discounted future reward suffers from high variance and hence could hinder learning. Using a k-step
    return estimator introduces some bias but reduces variance, yielding a beneficial trade-off."""
    total_timesteps = len(reward_trajectory)

    k_step_return_estimators = []
    for t in range(total_timesteps):
        combined_ksr_est = 0
        for k_curr in range(1, k + 1):
            ksr_est = 0
            for i in range(k_curr):
                if t + i < total_timesteps:
                    ksr_est += (discount_factor**i) * reward_trajectory[t + i]
            if t + k_curr < total_timesteps:
                ksr_est += (discount_factor**k_curr) * value_predictions[t + k_curr]

            combined_ksr_est += ksr_est * (gae_lambda ** (k_curr - 1))

        k_step_return_estimators.append(combined_ksr_est * (1 - gae_lambda))

    return k_step_return_estimators


if __name__ == "__main__":
    rewards = list(range(1000))
    returns = get_discounted_returns(rewards, 0.99)
    print(returns)
    print(generalized_advantage_estimator(
        rewards,
        returns,
        discount_factor=0.99,
        gae_lambda=0.95,
        k=3
    ))
