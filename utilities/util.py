#!/usr/bin/env python
"""Helper functions."""
import random
from typing import Tuple, Union, List, Dict

import gym
import numpy
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.spaces import Discrete, Box
from mpi4py import MPI
from tensorflow.python.client import device_lib

from utilities.error import UninterpretableObservationSpace


def get_available_gpus():
    """Get list of available GPUs."""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def mpi_flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\r{string}", end="")


def set_all_seeds(seed):
    """Set all random seeds (tf, np, random) to given value."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def env_extract_dims(env: gym.Env) -> Tuple[Dict[str, Tuple], int]:
    """Returns state and action space dimensionality for given environment."""
    obs_dim: dict

    # extract observation space dimensionalities from environment
    if isinstance(env.observation_space, spaces.Dict):
        # dict observation with observation field, for all GoalEnvs
        if isinstance(env.observation_space["observation"], spaces.Box):
            obs_dim = {"proprioception": env.observation_space["observation"].shape}
        elif isinstance(env.observation_space["observation"], spaces.Dict):
            # made for sensation
            obs_dim = {field: field.shape for field in env.observation_space["observation"]}
        else:
            raise UninterpretableObservationSpace(f"Cannot extract the dimensionality from a spaces.Dict observation space "
                                                  f"where the observation is of type "
                                                  f"{type(env.observation_space['observation']).__name__}")
    elif isinstance(env.observation_space, gym.spaces.Box):  # standard observation in box form
        obs_dim = {"proprioception": env.observation_space.shape}
    else:
        raise UninterpretableObservationSpace(
            f"Cannot interpret observation space of type {type(env.observation_space)}.")

    # action space
    if isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
    elif isinstance(env.action_space, Box):
        act_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Environment has unknown Action Space Typ: {env.action_space}")

    return obs_dim, act_dim


def normalize(x, is_img=False) -> numpy.ndarray:
    """Normalize a numpy array to have all values in range (0, 1)."""
    x = tf.convert_to_tensor(x).numpy()
    return x / 255 if is_img else (x - x.min()) / (x.max() - x.min())


def flatten(some_list):
    """Flatten a python list."""
    return [some_list] if not isinstance(some_list, list) else [x for X in some_list for x in flatten(X)]


def is_array_collection(a: numpy.ndarray) -> bool:
    """Check if an array is an array of objects (e.g. goal arrays) or an actual array of direct data."""
    return a.dtype == "O"


def add_state_dims(state: "Sensation", dims: int = 1, axis: int = 0) -> 'Sensation':
    """Expand state (array or lost of arrays) to have a batch and/or time dimension."""
    if dims < 1:
        return state

    new_state = state.with_leading_dims()

    return new_state


def merge_into_batch(list_of_states: List[Union[numpy.ndarray, Tuple]]):
    """Merge a list of states into one huge batch of states. Handles both single and multi input states.

    Assumes NO batch dimension!
    """
    if isinstance(list_of_states[0], numpy.ndarray):
        return numpy.concatenate(add_state_dims(list_of_states))
    else:
        return tuple(numpy.concatenate(list(map(lambda x: add_state_dims(x[i]), list_of_states)), axis=0)
                     for i in range(len(list_of_states[0])))


def insert_unknown_shape_dimensions(shape, none_replacer: int = 1):
    """Replace Nones in a shape tuple with 1 or a given goal value."""
    return tuple(map(lambda s: none_replacer if s is None else s, shape))


def detect_finished_episodes(action_log_probabilities: tf.Tensor):
    """Detect which samples in the batch connect to a episode that finished during the subsequence, based on the step_tuple
    log probabilities and return a 1D boolean tensor.

    Input Shape:
        action_probabilities: (B, S)
    """
    # TODO wont work for episodes that finish exactly at end of sequence
    # need to check only last one, as checking any might catch (albeit unlikely) true 0 in the sequence
    finished = action_log_probabilities[:, -1] == 0
    return finished


if __name__ == "__main__":
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
