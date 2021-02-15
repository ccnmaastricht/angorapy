import abc
import inspect
import sys
from collections import namedtuple
from typing import List, Union, Tuple, Iterable

import gym
import numpy as np

from utilities.const import NP_FLOAT_PREC, EPSILON
from utilities.util import env_extract_dims, parse_state

TransformerSerialization = namedtuple("TransformerSerialization", ["class_name", "data"])


class BaseTransformer(abc.ABC):
    """Abstract base class for preprocessors."""

    def __init__(self, env):
        self.n = 1e-4  # make this at least epsilon so that first measure is not all zeros
        self.previous_n = self.n

        # extract env info
        self.env = env
        self.state_dim, self.number_of_actions = env_extract_dims(env)

    @abc.abstractmethod
    def __add__(self, other):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @property
    def name(self):
        """The name of the transformers."""
        return self.__class__.__name__

    def transform(self, step_result, **kwargs):
        """Transform the step results."""
        pass

    def update(self, **kwargs):
        """Just for the sake of interchangeability, all transformers have an update method even if they do not update."""
        pass

    def warmup(self, env, n_steps=10):
        """Warm up the transformer for n steps."""
        pass

    def serialize(self) -> dict:
        """Serialize the transformer to allow for saving its data in a file."""
        return {self.__class__.__name__: (self.n,)}

    @staticmethod
    def from_serialization(serialization: TransformerSerialization):
        """Create transformer from a serialization."""
        return getattr(sys.modules[__name__], serialization.class_name).recover(serialization.data)

    @classmethod
    @abc.abstractmethod
    def recover(cls, serialization_data: list):
        """Recover from serialization."""
        pass

    @staticmethod
    def from_collection(collection_of_transformers):
        """Merge a list of transformers into one new transformer of the same type."""
        assert len(set([type(w) for w in collection_of_transformers])) == 1, \
            "All transformers need to have the same type."

        new_transformer = collection_of_transformers[0]
        for transformer in collection_of_transformers[1:]:
            new_transformer += transformer
        return new_transformer

    def correct_sample_size(self, deduction):
        """Deduce the given number from the sample counter."""
        self.n = self.n - deduction


class BaseRunningMeanTransformer(BaseTransformer, abc.ABC):
    """Abstract base class for transformers implementing a running mean over some statistic."""

    mean: List[np.ndarray]
    variance: List[np.ndarray]

    def __add__(self, other) -> "BaseRunningMeanTransformer":
        nt = self.__class__(self.env)
        nt.n = self.n + other.n

        for i in range(len(self.mean)):
            nt.mean[i] = (self.n / nt.n) * self.mean[i] + (other.n / nt.n) * other.mean[i]
            nt.variance[i] = (self.n * (self.variance[i] + (self.mean[i] - nt.mean[i]) ** 2)
                              + other.n * (other.variance[i] + (other.mean[i] - nt.mean[i]) ** 2)) / nt.n

        return nt

    def update(self, observation: Union[Tuple[np.ndarray], np.ndarray]) -> None:
        """Update the mean(serialization) and variance(serialization) of the tracked statistic based on the new sample.

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

    def serialize(self) -> TransformerSerialization:
        """Serialize the transformer to allow for saving it in a file."""
        return TransformerSerialization(self.__class__.__name__, [self.n,
                                                                  list(map(lambda m: m.tolist(), self.mean)),
                                                                  list(map(lambda v: v.tolist(), self.variance))])

    @classmethod
    def recover(cls, data):
        """Recover a running mean transformer from its serialization"""
        transformer = cls() if len(data) == 3 else cls(data[3])
        transformer.n = np.array(data[0])
        transformer.mean = list(map(lambda l: np.array(l), data[1]))
        transformer.variance = list(map(lambda l: np.array(l), data[2]))

        return transformer

    def simplified_mean(self) -> List[float]:
        """Get a simplified, one dimensional mean by meaning any means."""
        return [np.mean(m).item() for m in self.mean]

    def simplified_variance(self) -> List[float]:
        """Get a simplified, one dimensional variance by meaning any variances."""
        return [np.mean(v).item() for v in self.variance]

    def simplified_stdev(self) -> List[float]:
        """Get a simplified, one dimensional stdev by meaning any variances and taking their square root."""
        return [np.sqrt(np.mean(v)).item() for v in self.variance]


class StateNormalizationTransformer(BaseRunningMeanTransformer):
    """Transformer for state normalization using running mean and variance estimations."""

    def __init__(self, env):
        # parse input types into normed shape format
        super().__init__(env)

        self.shapes: Iterable[Iterable[int]]
        if isinstance(self.state_dim, Iterable) and all(isinstance(x, Iterable) for x in self.state_dim):
            self.shapes = self.state_dim
        elif isinstance(self.state_dim, int):
            self.shapes = ((self.state_dim,),)
        elif isinstance(self.state_dim, Iterable) and all(isinstance(x, int) for x in self.state_dim):
            self.shapes = (self.state_dim,)
        else:
            raise ValueError("Cannot understand shape format.")

        self.mean = [np.zeros(i_shape, NP_FLOAT_PREC) for i_shape in self.shapes if len(i_shape) == 1]
        self.variance = [np.ones(i_shape, NP_FLOAT_PREC) for i_shape in self.shapes if len(i_shape) == 1]

        assert len(self.mean) > 0 and len(self.variance) > 0, "Initialized StateNormalizationTransformer got no vector " \
                                                              "states."

    def step(self, step_result, update=True) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        o, r, done, info = step_result

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

    def warmup(self, env, n_steps=10):
        """Warmup the transformer by sampling the observation space."""
        env.reset()
        for i in range(n_steps):
            self.update(parse_state(env.observation_space.sample()))

    def serialize(self) -> TransformerSerialization:
        """Serialize the transformer to allow for saving it in a file."""
        serialization = super().serialize()
        serialization.data.append(self.shapes)

        return serialization


class RewardNormalizationTransformer(BaseRunningMeanTransformer):
    """Transformer for reward normalization using running mean and variance estimations."""

    def __init__(self, env):
        super().__init__(env)

        self.mean = [np.array(0, NP_FLOAT_PREC)]
        self.variance = [np.array(1, NP_FLOAT_PREC)]
        self.ret = np.float64(0)

    def transform(self, step_tuple: Tuple, update=True) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        o, r, done, info = step_tuple

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

    def warmup(self, env: gym.Env, n_steps=10):
        """Warmup the transformer by randomly stepping the environment through step_tuple space sampling."""
        env.reset()
        for i in range(n_steps):
            self.update(env.step(env.action_space.sample())[1])


class StateMemory(BaseTransformer):
    pass


def merge_transformers(transformers: List[BaseTransformer]) -> BaseTransformer:
    """Merge a list of transformers into a single transformer.

    Args:
        transformers:           list of transformers
    """
    assert all(type(t) is type(transformers[0]) for t in transformers), \
        "To merge transformers, they must be of same type."

    # merge the list of transformers into a single transformer
    merged_transformer = BaseTransformer.from_collection(transformers)

    # adjust for overcounting
    merged_transformer.correct_sample_size((len(transformers) - 1) * transformers[0].previous_n)

    # record the new n for next sync
    merged_transformer.previous_n = merged_transformer.n

    return merged_transformer
