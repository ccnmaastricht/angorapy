#!/usr/bin/env python
"""Helper functions."""
import itertools
import random
from typing import Tuple, Union, List

import gym
import numpy
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box, Dict
from tensorflow.python.client import device_lib


def get_available_gpus():
    """Get list of available GPUs."""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    print(f"\r{string}", end="")


def set_all_seeds(seed):
    """Set all random seeds (tf, np, random) to given value."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def env_extract_dims(env: gym.Env) -> Tuple[Union[int, tuple], int]:
    """Returns state and (discrete) action space dimensionality for given environment."""
    if isinstance(env.observation_space, Dict):
        obs_dim = tuple(field.shape for field in env.observation_space["observation"])
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


def is_recurrent_model(model: tf.keras.Model):
    """Check if given model is recurrent (i.e. contains a recurrent layer of any sort)"""
    for layer in extract_layers(model):
        if isinstance(layer, tf.keras.layers.RNN):
            return True

    return False


def is_array_collection(a: numpy.ndarray):
    """Check if an array is an array of objects (e.g. other arrays) or an actual array of direct data."""
    return a.dtype == "O"


def parse_state(state: Union[numpy.ndarray, dict]) -> Union[numpy.ndarray, Tuple]:
    """Parse a state (array or array of arrays) received from an environment to have type float32."""
    return state.astype(numpy.float32) if not isinstance(state, dict) else \
        tuple(map(lambda x: x.astype(numpy.float32), state["observation"]))


def add_state_dims(state: Union[numpy.ndarray, Tuple], dims: int = 1, axis: int = 0) -> Union[numpy.ndarray, Tuple]:
    """Expand state (array or lost of arrays) to have a batch dimension."""
    if dims < 1:
        return state

    return numpy.expand_dims(add_state_dims(state, dims=dims - 1, axis=axis), axis=axis) if not isinstance(state, Tuple) \
        else tuple(map(lambda x: numpy.expand_dims(x, axis=axis), add_state_dims(state, dims=dims - 1, axis=axis)))


def merge_into_batch(list_of_states: List[Union[numpy.ndarray, Tuple]]):
    """Merge a list of states into one huge batch of states. Handles both single and multi input states.

    Assumes NO batch dimension!
    """
    if isinstance(list_of_states[0], numpy.ndarray):
        return numpy.concatenate(add_state_dims(list_of_states))
    else:
        return tuple(numpy.concatenate(list(map(lambda x: add_state_dims(x[i]), list_of_states)), axis=0)
                     for i in range(len(list_of_states[0])))


def extract_layers(network: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    """Recursively extract layers from a potentially nested list of Sequentials of unknown depth."""
    return list(itertools.chain(*[extract_layers(layer)
                                  if isinstance(layer, tf.keras.Sequential)
                                  else [layer] for layer in network.layers]))


if __name__ == "__main__":
    pass
