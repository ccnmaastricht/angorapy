"""Wrapping utilities preprocessing an environment output."""
import abc
import inspect
import sys
from typing import Tuple, Iterable

import gym
import numpy as np

from utilities.const import EPS, NUMPY_FLOAT_PRECISION


class _Wrapper(abc.ABC):

    def __init__(self):
        self.n = 1e-4  # make this at least epsilon so that first measure is not all zeros

    @abc.abstractmethod
    def __add__(self, other):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def wrap_a_step(self, step_output):
        """Preprocess an environment output."""
        pass

    def update(self, **kwargs):
        """Just for the sake of interchangeability, all wrappers have an update method even if they do not update."""
        pass

    def warmup(self, env: gym.Env, observations=10):
        """Warm up the wrappers on an env for n steps."""
        pass

    def serialize(self) -> dict:
        """Serialize the wrapper to allow for saving it in a file."""
        return {self.__class__.__name__: (self.n,)}

    @staticmethod
    def from_serialization(s: dict):
        """Create (combi-)wrapper from a serialization."""
        if len(s) > 1:
            return CombiWrapper([getattr(sys.modules[__name__], k).recover(v) for k, v in s.items()])
        else:
            return getattr(sys.modules[__name__], list(s.keys())[0]).recover(list(s.values())[0])

    @classmethod
    @abc.abstractmethod
    def recover(cls, serialization_data: list):
        """Recover from serialization."""
        pass

    @staticmethod
    def from_collection(collection_of_wrappers):
        """Merge a list of wrappers into one new wrapper of the same type."""
        assert len(set([type(w) for w in collection_of_wrappers])) == 1, "All wrappers need to have the same type."

        new_wrapper = collection_of_wrappers[0]
        for wrapper in collection_of_wrappers[1:]:
            new_wrapper += wrapper
        return new_wrapper

    def correct_sample_size(self, deduction):
        """Deduce the given number from the sample counter."""
        self.n -= deduction


class _RunningMeanWrapper(_Wrapper, abc.ABC):
    mean: np.ndarray
    variance: np.ndarray

    def __init__(self, **args):
        super().__init__()

    def __add__(self, other) -> "_RunningMeanWrapper":
        needs_shape = len(inspect.signature(self.__class__).parameters) > 0
        new_wrapper = self.__class__(self.mean.shape) if needs_shape else self.__class__()
        new_wrapper.n = self.n + other.n
        new_wrapper.mean = (self.n / new_wrapper.n) * self.mean + (other.n / new_wrapper.n) * other.mean
        new_wrapper.variance = (self.n * (self.variance + (self.mean - new_wrapper.mean) ** 2)
                                + other.n * (other.variance + (other.mean - new_wrapper.mean) ** 2)) / new_wrapper.n

        return new_wrapper

    def update(self, observation):
        """Update the mean and variance of the tracked statistic based on the new sample.
        Simplification of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm"""
        self.n += 1

        delta = observation - self.mean
        m_a = self.variance * (self.n - 1)

        self.mean = np.array(self.mean + delta * (1 / self.n), dtype=NUMPY_FLOAT_PRECISION)
        self.variance = np.array((m_a + np.square(delta) * (self.n - 1) / self.n) / self.n, dtype=NUMPY_FLOAT_PRECISION)

    def serialize(self) -> dict:
        """Serialize the wrapper to allow for saving it in a file."""
        return {self.__class__.__name__: (self.n, self.mean.tolist(), self.variance.tolist())}

    @classmethod
    def recover(cls, serialization_data):
        """Recover a running mean wrapper from its serialization"""
        wrapper = cls() if len(serialization_data) == 3 else cls(serialization_data[3])
        wrapper.n = serialization_data[0]
        wrapper.mean = serialization_data[1]
        wrapper.variance = serialization_data[2]

        return wrapper


class StateNormalizationWrapper(_RunningMeanWrapper):
    """Wrapper for state normalization using running mean and variance estimations."""

    def __init__(self, state_shape):
        super().__init__()
        self.shape = state_shape
        self.mean = np.zeros(state_shape, NUMPY_FLOAT_PRECISION)
        self.variance = np.ones(state_shape, NUMPY_FLOAT_PRECISION)

    def wrap_a_step(self, step_result: Tuple) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        try:
            o, r, done, info = step_result
        except ValueError:
            raise ValueError("Wrapping did not receive a valid input.")

        self.update(o)

        # normalize
        o = np.clip((o - self.mean) / (np.sqrt(self.variance + EPS)), -10., 10.)
        return o, r, done, info

    def warmup(self, env: gym.Env, observations=10):
        """Warmup the wrapper by sampling the observation space."""
        for i in range(observations):
            self.update(env.observation_space.sample())

    def serialize(self) -> dict:
        """Serialize the wrapper to allow for saving it in a file."""
        serialization = super().serialize()
        serialization[self.__class__.__name__] += (self.shape,)

        return serialization


class RewardNormalizationWrapper(_RunningMeanWrapper):
    """Wrapper for reward normalization using running mean and variance estimations."""

    def __init__(self):
        super().__init__()
        self.mean = np.array(0, NUMPY_FLOAT_PRECISION)
        self.variance = np.array(1, NUMPY_FLOAT_PRECISION)
        self.ret = np.float64(0)

    def wrap_a_step(self, step_result: Tuple) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        try:
            o, r, done, info = step_result
        except ValueError:
            raise ValueError("Wrapping did not receive a valid input.")

        if r is None:
            return o, r, done, info  # skip

        # update based on cumulative discounted reward
        self.ret = 0.99 * self.ret + r
        self.update(self.ret)

        # normalize
        r = np.clip(r / (np.sqrt(self.variance + EPS)), -10., 10.)

        if done:
            self.ret = 0.

        return o, r, done, info

    def warmup(self, env: gym.Env, observations=10):
        """Warmup the wrapper by randomly stepping the environment through action space sampling."""
        env = gym.make(env.unwrapped.spec.id)  # make new to not interfere with original when stepping
        env.reset()
        for i in range(observations):
            self.update(env.step(env.action_space.sample())[1])


class SkipWrapper(_Wrapper):
    """Simple Passing Wrapper. Does nothing. Just for convenience."""

    @classmethod
    def recover(cls, serialization_data):
        """Recovery is just creation of new, no info to be recovered from."""
        return SkipWrapper()

    def wrap_a_step(self, step_output):
        """Wrap by doing nothing."""
        return step_output

    def __add__(self, other):
        return self


class CombiWrapper(_Wrapper):
    """Combine any number of arbitrary wrappers into one using this interface. Meaningless wrappers (SkipWrappers) will
    be automatically neglected."""

    def __init__(self, wrappers: Iterable[_Wrapper]):
        super().__init__()
        self.wrappers = [w for w in wrappers if not isinstance(w, SkipWrapper)]

    def __add__(self, other):
        added_wraps = CombiWrapper([self.wrappers[i] + other.wrappers[i] for i in range(len(self.wrappers))])
        added_wraps.n = self.n + other.n

        return added_wraps

    def __len__(self):
        return len(self.wrappers)

    def __getitem__(self, item):
        return self.wrappers[item]

    def __iter__(self):
        return self.wrappers

    def __repr__(self):
        return f"CombiWrapper{tuple(str(w) for w in self.wrappers)}"

    def wrap_a_step(self, step_output):
        """Wrap a step by passing it through all contained wrappers."""
        for w in self.wrappers:
            step_output = w.wrap_a_step(step_output)

        self.n += 1
        return step_output

    def correct_sample_size(self, deduction):
        """Correct the sample size for all wrappers in this combi."""
        super().correct_sample_size(deduction)
        for wrapper in self.wrappers:
            wrapper.correct_sample_size(deduction)

    def warmup(self, env: gym.Env, observations=10):
        """Warm up all contained wrappers."""
        for w in self.wrappers:
            w.warmup(env, observations=observations)

    def serialize(self) -> dict:
        """Serialize all wrappers in the combi into one representation."""
        full = {}
        for w in self.wrappers:
            full.update(w.serialize())

        return full

    def recover(cls, serialization_data):
        """Recover CombiWrapper, not supported."""
        raise NotImplementedError("There is no such thing as recovery for CombiWrappers. Create a new wrapper from"
                                  "recovered wrappers instead.")
