"""Wrappers encapsulating envs to modulate n_steps, rewards, and control state initialization."""
import abc
from typing import List, OrderedDict, Tuple, Dict, Any, SupportsFloat

import gymnasium as gym
import numpy

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from angorapy.common.senses import Sensation
from angorapy.common.postprocessors import BasePostProcessor, merge_postprocessors


class TaskWrapper(gym.ObservationWrapper):
    """Wrapper for all tasks with basic functionality.

    Parses observations of the wrapped task to Sensations. Serves additionally as base class for all other wrappers.

    Args:
        env (gym.Env): The task to wrap.

    Attributes:
        env (gym.Env): The wrapped environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    @property
    def name(self):
        """The name of the wrapper."""
        return self.__class__.__name__

    @abc.abstractmethod
    def warmup(self, n_steps=10):
        """Warmup the environment."""
        pass

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), self.info(info)

    def observation(self, observation) -> Sensation:
        """Process an observation to be of type 'Sensation'."""
        if isinstance(observation, Sensation):
            return observation
        elif isinstance(observation, (dict, OrderedDict)):
            assert "observation" in observation.keys(), "Unknown dict type of state couldnt be resolved to Sensation."

            if isinstance(observation["observation"], Sensation):
                return observation["observation"]
            elif isinstance(observation["observation"], numpy.ndarray):  # GOAl ENVS
                return Sensation(proprioception=observation["observation"], goal=observation["desired_goal"])
            elif isinstance(observation["observation"], dict) and all(
                    [k in observation["observation"].keys() for k in Sensation.sense_names]):
                return Sensation(**observation["observation"])

        return Sensation(proprioception=observation)

    def step(self, action) -> Tuple[Sensation, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        if hasattr(observation, "keys") and "achieved_goal" in observation.keys():
            info["achieved_goal"] = observation["achieved_goal"]

        if hasattr(observation, "keys") and "desired_goal" in observation.keys():
            info["desired_goal"] = observation["desired_goal"]

        return self.observation(observation), reward, terminated, truncated, info

    def info(self, info):
        return info

    # SYNCHRONIZATION

    def mpi_sync(self):
        """Synchronize the wrapper and all wrapped wrappers over all MPI ranks."""
        if isinstance(self.env, TaskWrapper):
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


class PostProcessingWrapper(TaskWrapper):
    """Wrapper transforming rewards and observation based on running means."""

    def __init__(self, env, postprocessors: List[BasePostProcessor]):
        super().__init__(env)

        # TODO maybe change this to expect list of types to build itself?
        self.postprocessors = postprocessors

    def __contains__(self, item):
        return item in self.postprocessors

    def step(self, action):
        """Perform a step and transform the results."""
        step_tuple = super().step(action)

        # include original reward in info
        step_tuple[-1]["original_reward"] = step_tuple[1]

        if len(self.postprocessors) != 0:
            for postprocessor in self.postprocessors:
                step_tuple = postprocessor.transform(step_tuple, update=True)

        return step_tuple

    def add_postprocessors(self, postprocessors):
        """Add a list of postprocessors to the environment."""
        self.postprocessors.extend(postprocessors)

    def clear_postprocessors(self):
        """Clear the list of postprocessors."""
        self.postprocessors = []

    def warmup(self, n_steps=10):
        """Warmup the postprocessors."""
        for t in self.postprocessors:
            t.warmup(self, n_steps=n_steps)

    def _mpi_sync(self):
        """Synchronise the postprocessors of the wrapper over all MPI ranks."""
        if MPI is None:
            return

        synced_postprocessors = []
        for postprocessor in self.postprocessors:
            collection = MPI.COMM_WORLD.gather(postprocessor, root=0)

            if MPI.COMM_WORLD.Get_rank() == 0:
                synced_postprocessors.append(merge_postprocessors(collection))

        self.postprocessors = MPI.COMM_WORLD.bcast(synced_postprocessors, root=0)

    def serialize(self):
        """Return separate postprocessor serializations in a list"""
        return [t.serialize() for t in self.postprocessors]
