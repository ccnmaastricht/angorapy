"""Wrapping utilities preprocessing an environment output."""
import abc
import inspect
import itertools
import sys
from copy import copy
from typing import Tuple, Iterable, Union, List

import gym
import numpy as np
from mpi4py import MPI

from utilities.const import EPSILON, NP_FLOAT_PREC
from utilities.util import parse_state


class BaseWrapper(abc.ABC):
    """Abstract base class for preprocessors."""

    def __init__(self):
        self.n = 1e-4  # make this at least epsilon so that first measure is not all zeros

    @abc.abstractmethod
    def __add__(self, other):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def __iter__(self):
        return iter([self])

    def __contains__(self, item):
        if isinstance(item, BaseWrapper):
            return item.name == self.name
        elif isinstance(item, str):
            return item == self.name
        elif isinstance(item, type):
            return item == self.__class__
        else:
            return False

    @property
    def name(self):
        """The name of the preprocessor."""
        return self.__class__.__name__

    @abc.abstractmethod
    def modulate(self, step_output, update=True):
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
            combi_wrappi = CombiWrapper([getattr(sys.modules[__name__], k).recover(v) for k, v in s.items()])
            combi_wrappi.n = copy(combi_wrappi[0].n)
            return combi_wrappi
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
        self.n = self.n - deduction


class BaseRunningMeanWrapper(BaseWrapper, abc.ABC):
    """Abstract base class for wrappers implementing a running mean over some statistic."""

    mean: List[np.ndarray]
    variance: List[np.ndarray]

    def __init__(self, **args):
        super().__init__()

    def __add__(self, other) -> "BaseRunningMeanWrapper":
        needs_shape = len(inspect.signature(self.__class__).parameters) > 0
        nw = self.__class__(tuple(m.shape for m in self.mean)) if needs_shape else self.__class__()
        nw.n = self.n + other.n

        for i in range(len(self.mean)):
            nw.mean[i] = (self.n / nw.n) * self.mean[i] + (other.n / nw.n) * other.mean[i]
            nw.variance[i] = (self.n * (self.variance[i] + (self.mean[i] - nw.mean[i]) ** 2)
                              + other.n * (other.variance[i] + (other.mean[i] - nw.mean[i]) ** 2)) / nw.n

        return nw

    def update(self, observation: Union[Tuple[np.ndarray], np.ndarray]) -> None:
        """Update the mean(s) and variance(s) of the tracked statistic based on the new sample.

        Simplification of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.

        The method can handle multi input states where the observation is a tuple of numpy arrays. For each element
        a separate mean/variance is tracked. Any non-vector input will be skipped as they are assumed to be images
        and should be handled seperatly."""
        self.n += 1

        if not isinstance(observation, Tuple):
            observation = (observation,)

        for i, obs in enumerate(filter(lambda o: isinstance(o, (int, float)) or len(o.shape) in [0, 1], observation)):
            delta = obs - self.mean[i]
            m_a = self.variance[i] * (self.n - 1)

            self.mean[i] = np.array(self.mean[i] + delta * (1 / self.n), dtype=NP_FLOAT_PREC)
            self.variance[i] = np.array((m_a + np.square(delta) * (self.n - 1) / self.n) / self.n, dtype=NP_FLOAT_PREC)

    def serialize(self) -> dict:
        """Serialize the wrapper to allow for saving it in a file."""
        return {self.__class__.__name__: (self.n,
                                          list(map(lambda m: m.tolist(), self.mean)),
                                          list(map(lambda v: v.tolist(), self.variance))
                                          )}

    @classmethod
    def recover(cls, serialization_data):
        """Recover a running mean wrapper from its serialization"""
        wrapper = cls() if len(serialization_data) == 3 else cls(serialization_data[3])
        wrapper.n = np.array(serialization_data[0])
        wrapper.mean = list(map(lambda l: np.array(l), serialization_data[1]))
        wrapper.variance = list(map(lambda l: np.array(l), serialization_data[2]))

        return wrapper

    def simplified_mean(self) -> List[float]:
        """Get a simplified, one dimensional mean by meaning any means."""
        return [np.mean(m).item() for m in self.mean]

    def simplified_variance(self) -> List[float]:
        """Get a simplified, one dimensional variance by meaning any variances."""
        return [np.mean(v).item() for v in self.variance]

    def simplified_stdev(self) -> List[float]:
        """Get a simplified, one dimensional stdev by meaning any variances and taking their square root."""
        return [np.sqrt(np.mean(v)).item() for v in self.variance]


class StateNormalizationWrapper(BaseRunningMeanWrapper):
    """Wrapper for state normalization using running mean and variance estimations."""

    def __init__(self, state_shapes: Union[Iterable[Iterable[int]], Iterable[int], int]):
        super().__init__()

        # parse input types into normed shape format
        self.shapes: Iterable[Iterable[int]]
        if isinstance(state_shapes, Iterable) and all(isinstance(x, Iterable) for x in state_shapes):
            self.shapes = state_shapes
        elif isinstance(state_shapes, int):
            self.shapes = ((state_shapes,),)
        elif isinstance(state_shapes, Iterable) and all(isinstance(x, int) for x in state_shapes):
            self.shapes = (state_shapes,)
        else:
            raise ValueError("Cannot understand shape format.")

        self.mean = [np.zeros(i_shape, NP_FLOAT_PREC) for i_shape in self.shapes if len(i_shape) == 1]
        self.variance = [np.ones(i_shape, NP_FLOAT_PREC) for i_shape in self.shapes if len(i_shape) == 1]

        assert len(self.mean) > 0 and len(self.variance) > 0, "Initialized StateNormalizationWrapper got no vector " \
                                                              "states."

    def modulate(self, step_result: Tuple, update=True) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        try:
            o, r, done, info = step_result
        except ValueError:
            raise ValueError("Wrapping did not receive a valid input.")

        if update:
            self.update(o)

        # normalize
        if not isinstance(o, Tuple):
            o = (o,)

        normed_o = []
        for i, op in enumerate(filter(lambda a: len(a.shape) == 1, o)):
            normed_o.append(np.clip((op - self.mean[i]) / (np.sqrt(self.variance[i] + EPSILON)), -10., 10.))

        normed_o = normed_o[0] if len(normed_o) == 1 else tuple(normed_o)
        return normed_o, r, done, info

    def warmup(self, env: gym.Env, observations=10):
        """Warmup the wrapper by sampling the observation space."""
        for i in range(observations):
            self.update(parse_state(env.observation_space.sample()))

    def serialize(self) -> dict:
        """Serialize the wrapper to allow for saving it in a file."""
        serialization = super().serialize()
        serialization[self.__class__.__name__] += (self.shapes,)

        return serialization


class RewardNormalizationWrapper(BaseRunningMeanWrapper):
    """Wrapper for reward normalization using running mean and variance estimations."""

    def __init__(self):
        super().__init__()
        self.mean = [np.array(0, NP_FLOAT_PREC)]
        self.variance = [np.array(1, NP_FLOAT_PREC)]
        self.ret = np.float64(0)

    def modulate(self, step_result: Tuple, update=True) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        try:
            o, r, done, info = step_result
        except ValueError:
            raise ValueError("Wrapping did not receive a valid input.")

        if r is None:
            return o, r, done, info  # skip

        # update based on cumulative discounted reward
        if update:
            self.ret = 0.99 * self.ret + r
            self.update(self.ret)

        # normalize
        r = np.clip(r / (np.sqrt(self.variance[0] + EPSILON)), -10., 10.)

        if done:
            self.ret = 0.

        return o, r, done, info

    def warmup(self, env: gym.Env, observations=10):
        """Warmup the wrapper by randomly stepping the environment through action space sampling."""
        env = gym.make(env.unwrapped.spec.id)  # make new to not interfere with original when stepping
        env.reset()
        for i in range(observations):
            self.update(env.step(env.action_space.sample())[1])


class SkipWrapper(BaseWrapper):
    """Simple Passing Wrapper. Does nothing. Just for convenience."""

    @classmethod
    def recover(cls, serialization_data):
        """Recovery is just creation of new, no info to be recovered from."""
        return SkipWrapper()

    def modulate(self, step_output, update=True):
        """Wrap by doing nothing."""
        return step_output

    def __add__(self, other):
        return self


class CombiWrapper(BaseWrapper, object):
    """Combine any number of arbitrary wrappers into one using this interface. Meaningless wrappers (SkipWrappers) will
    be automatically neglected."""

    def __init__(self, wrappers: Iterable[BaseWrapper]):
        super().__init__()
        self.wrappers = [w for w in wrappers if not isinstance(w, SkipWrapper)]
        if len(self.wrappers) == 0:
            self.__class__ = SkipWrapper  # not beautiful, but __new__ does not work with pickle for some reason

    def __add__(self, other):
        added_wraps = CombiWrapper([self.wrappers[i] + other.wrappers[i] for i in range(len(self.wrappers))])
        added_wraps.n = copy(self.n) + copy(other.n)

        return added_wraps

    def __contains__(self, item):
        return any([item in w for w in self.wrappers])

    def __len__(self):
        return len(self.wrappers)

    def __getitem__(self, item):
        return self.wrappers[item]

    def __iter__(self):
        return iter(self.wrappers)

    def __repr__(self):
        return f"CombiWrapper{tuple(str(w) for w in self.wrappers)}"

    def modulate(self, step_output, update=True):
        """Wrap a step by passing it through all contained wrappers."""
        for w in self.wrappers:
            step_output = w.modulate(step_output, update=update)

        if update:
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

        self.n += observations

    def serialize(self) -> dict:
        """Serialize all wrappers in the combi into one representation."""
        full = {}
        for w in self.wrappers:
            full.update(w.serialize())

        return full

    @classmethod
    def recover(cls, serialization_data):
        """Recover CombiWrapper, not supported."""
        raise NotImplementedError("There is no such thing as recovery for CombiWrappers. Create a new wrapper from"
                                  "recovered wrappers instead.")


def merge_wrappers(wrappers: List[BaseWrapper], old_wrapper_state: BaseWrapper) -> BaseWrapper:
    """Merge a list of wrappers into a single wrapper.

    Args:
        wrappers:           list of wrappers
        old_wrapper_state:  previous wrapper state before gathering
    """
    old_ns = [w.n for w in old_wrapper_state]
    old_n = old_wrapper_state.n

    merged_wrapper = BaseWrapper.from_collection(wrappers)

    for i, p in enumerate(merged_wrapper):
        p.correct_sample_size((len(wrappers) - 1) * old_ns[i])  # adjust for overcounting

    if isinstance(merged_wrapper, CombiWrapper):
        merged_wrapper.n = np.copy(merged_wrapper.n) - (len(wrappers) - 1) * old_n

    return merged_wrapper


def mpi_merge_wrappers(wrappers: List[BaseWrapper], old_wrapper_state: BaseWrapper) -> BaseWrapper:
    """Merge the wrappers of all MPI workers (and root) into a single wrapper.

    Args:
        wrappers:           buffer object of the wrappers in each worker
        old_wrapper_state:  previous wrapper state before gathering
    """
    collection = MPI.COMM_WORLD.gather(wrappers, root=0)

    if MPI.COMM_WORLD.Get_rank() == 0:
        collection = list(itertools.chain(*collection))
        return merge_wrappers(collection, old_wrapper_state)


if __name__ == '__main__':
    a = CombiWrapper([RewardNormalizationWrapper(), StateNormalizationWrapper(10)])
    b = CombiWrapper([RewardNormalizationWrapper(), StateNormalizationWrapper(10)])
    c = CombiWrapper([RewardNormalizationWrapper(), StateNormalizationWrapper(10)])

    for i in range(10):
        a.modulate([np.random.randn(10), np.random.randint(-5, 5), None, None])
        b.modulate([np.random.randn(10), np.random.randint(-5, 5), None, None])
        c.modulate([np.random.randn(10), np.random.randint(-5, 5), None, None])

    print(a.n, b.n, c.n)

    d = BaseWrapper.from_collection([a, b, c])

    print(d.n)
    print([w.n for w in d])

    a, b, c = BaseWrapper.from_serialization(a.serialize()), \
              BaseWrapper.from_serialization(b.serialize()), \
              BaseWrapper.from_serialization(c.serialize())

    print(a.n, b.n, c.n)

    old_n = 2
    d = BaseWrapper.from_collection([a, b, c])

    print(d.n)
    print([w.n for w in d])

    d.correct_sample_size((3 - 1) * old_n)  # adjust for overcounting

    print(d.n)
    print([w.n for w in d])
