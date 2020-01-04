"""Wrapping utilities preprocessing an environment output."""
import abc
from typing import Tuple

import numpy as np


class _Wrapper(abc.ABC):

    @abc.abstractmethod
    def wrap_a_step(self, step_output):
        """Preprocess an environment output."""
        pass

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @staticmethod
    def from_collection(collection_of_wrappers):
        """Merge a list of wrappers into one new wrapper of the same type."""
        new_wrapper = collection_of_wrappers[0]
        for wrapper in collection_of_wrappers[1:]:
            new_wrapper += wrapper
        return new_wrapper


class SkipWrapper(_Wrapper):

    def wrap_a_step(self, step_output):
        return step_output

    def __add__(self, other):
        return self


class StateNormalizationWrapper(_Wrapper):
    mu: np.ndarray
    variance: np.ndarray

    def __init__(self, state_shape):
        self.n = 0
        self.mu = np.zeros(state_shape)
        self.variance = np.ones(state_shape)

    def __add__(self, other):
        new_wrapper = StateNormalizationWrapper(self.mu.shape)
        new_wrapper.n = self.n + other.n
        new_wrapper.mu = (self.n / new_wrapper.n) * self.mu + (other.n / new_wrapper.n) * other.mu
        new_wrapper.variance = (self.n * (self.variance + (self.mu - new_wrapper.mu) ** 2)
                                + other.n * (other.variance + (other.mu - new_wrapper.mu) ** 2)) / new_wrapper.n

        return new_wrapper

    def wrap_a_step(self, step_result: Tuple) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""

        try:
            o, r, done, info = step_result
        except ValueError:
            raise ValueError("Wrapping did not receive a valid input.")

        mu_old = self.mu.copy()

        # calculate weights based on number of seen examples
        weight_experience = self.n / (self.n + 1)
        weight_state = 1 - weight_experience

        # update statistics
        self.n += 1
        self.mu = np.multiply(weight_experience, self.mu) + np.multiply(weight_state, o)
        self.variance = ((self.n - 1) * self.variance + (o - self.mu) * (o - mu_old)) / self.n

        # normalize
        o = (o - self.mu) / (np.sqrt(self.variance) + 1e-8)

        return o, r, done, info
