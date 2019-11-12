#!/usr/bin/env python
"""Helper functions."""
import random
from typing import Tuple, Union

import gym
import numpy
from gym.spaces import Discrete, Box, Dict
import tensorflow as tf
import numpy as np


def flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    print(f"\r{string}", end="")


def set_all_seeds(seed):
    """Set all random seeds (tf, np, random) to given value."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def env_extract_dims(env: gym.Env) -> Tuple[int, int]:
    """Returns state and (discrete) action space dimensionality for given environment."""
    if isinstance(env.observation_space, Dict):
        obs_dim = sum(field.shape[0] for field in env.observation_space["observation"])
    else:
        obs_dim = env.observation_space.shape[0]

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
    return x / 255 if not is_img else (x - x.min()) / (x.max() - x.min())


def flatten(some_list):
    """Flatten a python list."""
    return [some_list] if not isinstance(some_list, list) else [x for X in some_list for x in flatten(X)]


def is_array_collection(a: numpy.ndarray):
    """Check if an array is an array of objects (e.g. other arrays) or an actual array of direct data."""
    return a.dtype == "O"


def parse_state(state: Union[numpy.ndarray, dict]) -> Union[numpy.ndarray, Tuple]:
    """Parse a state (array or list of arrays) received from an environment to have type float32."""
    return state.astype(numpy.float32) if not is_array_collection(state) else \
        tuple(map(lambda x: x.astype(numpy.float32), state["observation"]))


def batchify_state(state: Union[numpy.ndarray, Tuple]) -> Union[numpy.ndarray, Tuple]:
    """Expand state (array or lost of arrays) to have a batch dimension."""
    return numpy.expand_dims(state, axis=0) if not is_array_collection(state) else \
        tuple(map(lambda x: numpy.expand_dims(x, axis=0), state))
