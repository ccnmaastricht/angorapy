"""Wrappers encapsulating environments to modulate observations, rewards, and control state initialization."""
import abc
import itertools
import sys
from copy import copy
from typing import Union, List, Type

import gym
import numpy as np
from mpi4py import MPI

from common.transformers import StateNormalizationTransformer, RewardNormalizationTransformer, BaseTransformer, \
    merge_transformers


class BaseWrapper(gym.Wrapper, abc.ABC):
    """Abstract base class for preprocessors."""

    def __init__(self, env):
        super().__init__(env)

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
        """The name of the transformers."""
        return self.__class__.__name__

    def update(self, **kwargs):
        """Just for the sake of interchangeability, all transformers have an update method even if they do not update."""
        pass

    def warmup(self, observations=10):
        """Warm up the wrapper for n steps."""
        pass

    def serialize(self) -> dict:
        """Serialize the wrapper to allow for saving its data in a file."""
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
        """Merge a list of transformers into one new wrapper of the same type."""
        assert len(set([type(w) for w in collection_of_wrappers])) == 1, "All transformers need to have the same type."

        new_wrapper = collection_of_wrappers[0]
        for wrapper in collection_of_wrappers[1:]:
            new_wrapper += wrapper
        return new_wrapper

    def correct_sample_size(self, deduction):
        """Deduce the given number from the sample counter."""
        self.n = self.n - deduction


class TransformationWrapper(BaseWrapper):

    def __init__(self, env, transformers: List[BaseTransformer]):
        super().__init__(env)

        self.transformers = transformers

    def add_transformers(self, transformers):
        """Add a list of transformers to the environment."""
        self.transformers.append(transformers)

    def clear_transformers(self):
        """Clear the list of transformers."""
        self.transformers = []

    def mpi_sync_transformers(self):
        """Synchronise the transformers of the wrapper over all MPI ranks."""
        for transformer in self.transformers:
            collection = MPI.COMM_WORLD.gather(transformer, root=0)

            if MPI.COMM_WORLD.Get_rank() == 0:
                collection = list(itertools.chain(*collection))
                return merge_transformers(collection, old_wrapper_state)


def make_env(env_name, reward_config: Union[str, dict] = None, transformers: List[BaseWrapper] = None) -> gym.Env:
    """Make environment, including a possible reward config and transformers."""
    if transformers is None:
        transformers = []

    env = TransformationWrapper(gym.make(env_name), transformers=transformers)

    if reward_config is not None and hasattr(env, "reward_config"):
        env.set_reward_function(reward_config)
        env.set_reward_config(reward_config)

    return env


if __name__ == '__main__':
    a = CombiWrapper([RewardNormalizationTransformer(), StateNormalizationTransformer(10)])
    b = CombiWrapper([RewardNormalizationTransformer(), StateNormalizationTransformer(10)])
    c = CombiWrapper([RewardNormalizationTransformer(), StateNormalizationTransformer(10)])

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
