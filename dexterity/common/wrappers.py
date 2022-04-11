"""Wrappers encapsulating environments to modulate n_steps, rewards, and control state initialization."""
import abc
from pprint import pprint
from typing import Union, List, Type, OrderedDict

import gym
import numpy
from mpi4py import MPI

from dexterity.common.senses import Sensation
from dexterity.common.transformers import BaseTransformer, merge_transformers
from dexterity.utilities.util import env_extract_dims


class BaseWrapper(gym.ObservationWrapper, abc.ABC):
    """Abstract base class for preprocessors."""

    def __init__(self, env):
        super().__init__(env)

    @property
    def name(self):
        """The name of the transformers."""
        return self.__class__.__name__

    @abc.abstractmethod
    def warmup(self, n_steps=10):
        """Warmup the environment."""
        pass

    def observation(self, observation):
        """Process an observation to be of type 'Sensation'."""
        if isinstance(observation, Sensation):
            return observation
        elif isinstance(observation, (dict, OrderedDict)):
            assert "observation" in observation.keys(), "Unknown dict type of state couldnt be resolved to Sensation."

            if isinstance(observation["observation"], Sensation):
                return observation["observation"]
            elif isinstance(observation["observation"], numpy.ndarray):  # GOAl ENVS
                return Sensation(proprioception=observation["observation"], goal=observation["desired_goal"])
            elif isinstance(observation["observation"], dict) and all([k in observation["observation"].keys() for k in Sensation.sense_names]):
                return Sensation(**observation["observation"])

        return Sensation(proprioception=observation)

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

    def step(self, action):
        """PErform a step and transform the results."""
        step_tuple = super().step(action)
        # include original reward in info
        step_tuple[-1]["original_reward"] = step_tuple[1]

        if len(self.transformers) != 0:
            for transformer in self.transformers:
                step_tuple = transformer.transform(step_tuple, update=True)

        return step_tuple

    def add_transformers(self, transformers):
        """Add a list of transformers to the environment."""
        self.transformers.extend(transformers)

    def clear_transformers(self):
        """Clear the list of transformers."""
        self.transformers = []

    def warmup(self, n_steps=10):
        """Warmup the transformers."""
        for t in self.transformers:
            t.warmup(self, n_steps=n_steps)

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


def make_env(env_name,
             reward_config: Union[str, dict] = None,
             transformers: List[Union[Type[BaseTransformer], BaseTransformer]] = None) -> BaseWrapper:
    """Make environment, including a possible reward config and transformers."""
    base_env = gym.make(env_name)
    state_dim, n_actions = env_extract_dims(base_env)

    if transformers is None:
        transformers = []
    elif all(isinstance(t, BaseTransformer) for t in transformers):
        transformers = transformers
    elif all(callable(t) for t in transformers):
        transformers = [t(env_name, state_dim, n_actions) for t in transformers]

    env = TransformationWrapper(base_env, transformers=transformers)

    if reward_config is not None and hasattr(env, "reward_config"):
        env.set_reward_function(reward_config)
        env.set_reward_config(reward_config)

    return env
