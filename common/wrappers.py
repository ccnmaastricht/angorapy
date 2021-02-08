"""Wrappers encapsulating environments to modulate n_steps, rewards, and control state initialization."""
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

    @property
    def name(self):
        """The name of the transformers."""
        return self.__class__.__name__

    @abc.abstractmethod
    def warmup(self, env, n_steps=10):
        """Warmup the environment."""
        pass

    # SYNCHRONIZATION

    def mpi_sync(self):
        """Synchronize the wrapper and all wrapped wrappers over all MPI ranks."""
        if isinstance(self.env, BaseWrapper):
            self.env.mpi_sync()

        self._mpi_sync()

    @abc.abstractmethod
    def _mpi_sync(self):
        pass

    # SERIALIZATION

    @abc.abstractmethod
    def serialize(self):
        """Serialize the wrappers defining data."""
        pass


class TransformationWrapper(BaseWrapper):
    """Wrapper transforming rewards and observation based on running means."""

    def __init__(self, env, transformers: List[BaseTransformer]):
        super().__init__(env)

        # TODO maybe change this to expect list of types to build itself?
        self.transformers = transformers

    def __contains__(self, item):
        return item in self.transformers

    def add_transformers(self, transformers):
        """Add a list of transformers to the environment."""
        self.transformers.extend(transformers)

    def clear_transformers(self):
        """Clear the list of transformers."""
        self.transformers = []

    def warmup(self, env, n_steps=10):
        """Warmup the transformers."""
        for t in self.transformers:
            t.warmup(env, n_steps=n_steps)

    def _mpi_sync(self):
        """Synchronise the transformers of the wrapper over all MPI ranks."""
        synced_transformers = []
        for transformer in self.transformers:
            collection = MPI.COMM_WORLD.gather(transformer, root=0)

            if MPI.COMM_WORLD.Get_rank() == 0:
                 synced_transformers.append(merge_transformers(collection))

        self.transformers = MPI.COMM_WORLD.bcast(synced_transformers, root=0)

    def serialize(self):
        """Return separate transformer serializations in a list"""
        return [t.serialize() for t in self.transformers]


def make_env(env_name, reward_config: Union[str, dict] = None, transformers: List[Type[BaseTransformer]] = None) \
        -> BaseWrapper:
    """Make environment, including a possible reward config and transformers."""
    if transformers is None:
        transformers = []

    base_env = gym.make(env_name)
    env = TransformationWrapper(base_env, transformers=[t(base_env) for t in transformers])

    if reward_config is not None and hasattr(env, "reward_config"):
        env.set_reward_function(reward_config)
        env.set_reward_config(reward_config)

    return env
