"""Helper functions."""
import random
import sys
from typing import Tuple, Union, List, Dict
import os

import gym
import numpy
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.spaces import Discrete, Box, MultiDiscrete
from mpi4py import MPI
from tensorflow.python.client import device_lib

from angorapy.utilities.error import UninterpretableObservationSpace


def get_available_gpus():
    """Get list of available GPUs."""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def mpi_flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\r{string}", end="")


def mpi_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(string)


def set_all_seeds(seed):
    """Set all random seeds (tf, np, random) to given value."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def env_extract_dims(env: gym.Env) -> Tuple[Dict[str, Tuple], Tuple[int]]:
    """Returns state and action space dimensionality for given environment."""
    obs_dim: dict

    # extract observation space dimensionalities from environment
    if isinstance(env.observation_space, spaces.Dict):
        # dict observation with observation field, for all GoalEnvs
        if isinstance(env.observation_space["observation"], spaces.Box):
            obs_dim = {"proprioception": env.observation_space["observation"].shape, "goal": env.observation_space["desired_goal"].shape}
        elif isinstance(env.observation_space["observation"], spaces.Dict):
            # made for sensation
            obs_dim = {name: field.shape for name, field in env.observation_space["observation"].spaces.items()}
        else:
            raise UninterpretableObservationSpace(f"Cannot extract the dimensionality from a spaces.Dict observation space "
                                                  f"where the observation is of type "
                                                  f"{type(env.observation_space['observation']).__name__}")
    elif isinstance(env.observation_space, gym.spaces.Box):  # standard observation in box form
        obs_dim = {"proprioception": env.observation_space.shape}
    elif isinstance(env.observation_space, gym.spaces.tuple.Tuple):
        obs_dim = {"proprioception": (len(env.observation_space),)}
    else:
        raise UninterpretableObservationSpace(
            f"Cannot interpret observation space of type {type(env.observation_space)}.")

    # action space
    if isinstance(env.action_space, Discrete):
        act_dim = (env.action_space.n,)
    elif isinstance(env.action_space, MultiDiscrete):
        assert np.alltrue(env.action_space.nvec == env.action_space.nvec[0]), "Can only handle multi-discrete action" \
                                                                              "spaces where all actions have the same " \
                                                                              "number of categories."
        act_dim = (env.action_space.shape[0], env.action_space.nvec[0].item())
    elif isinstance(env.action_space, Box):
        act_dim = (env.action_space.shape[0], 1)
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

    new_state = state.with_leading_dims(time=dims == 2)

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


def detect_finished_episodes(dones: tf.Tensor):
    """Detect which samples in the batch connect to a episode that finished during the subsequence, based on the dones
     and return a 1D boolean tensor.

    Input Shape:
        dones: (B, S)
    """
    finished = tf.math.reduce_any(dones, axis=-1)
    return finished


def find_divisors(number: int):
    divisors = []
    for i in range(1, number // 2 + 1):
        if number % i == 0:
            divisors.append(i)
            divisors.append(number // i)

    return list(sorted(set(divisors))) + [number]


def find_optimal_tile_shape(floor_shape: Tuple[int, int], tile_size: int, width_first=False) -> Tuple[int, int]:
    """For a given shape of a matrix (floor), find the shape of a tiles that fit the floor and contain
    exactly tile_size elements."""
    height_divisors = find_divisors(floor_shape[0])
    width_divisors = find_divisors(floor_shape[1])

    if not width_first:
        height_divisors = list(reversed(height_divisors))
    else:
        width_divisors = list(reversed(width_divisors))

    for hd in height_divisors:
        for wd in width_divisors:
            if hd * wd == tile_size:
                return hd, wd

    raise ValueError(f"No tiling of size {tile_size} possible for a floor of shape {floor_shape}.")


def stack_dicts(dicts: List[Dict[str, tf.Tensor]]):
    """Stack matching elements of a dict over a prepended domain."""
    return {
        key: np.stack([s[key] for s in dicts], axis=0) for key in dicts[0].keys()
    }


class HiddenPrints:
    """Context that hides print calls."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.
    The model shapes are multipled by the batch size, but the weights are not.
    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.
    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory


if __name__ == "__main__":
    from angorapy.configs import hp_config

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    conf = hp_config.manipulate
    # conf["workers"] = 8
    # conf["batch_size"] = 256

    n_optimizers = 32

    for wf in [True, False]:
        optimizer_tiling = find_optimal_tile_shape(
            (conf["workers"] // n_optimizers, conf["horizon"] // conf["tbptt"] // n_optimizers),
            tile_size=conf["batch_size"] // conf["tbptt"] // n_optimizers,
            width_first=wf
        )

        print(f"Optimizer's Tiling: {optimizer_tiling} with {['width', 'height'][wf]} first")