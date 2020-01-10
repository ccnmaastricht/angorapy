#!/usr/bin/env python
"""Helper functions."""
import random
from typing import Tuple, Union, List

import gym
import numpy
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box, Dict
from tensorflow.keras.layers import TimeDistributed
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


def list_layer_names(network, only_para_layers=True) -> List[str]:
    """Get a list of unique string representations of all layers in the network."""
    if only_para_layers:
        return [layer.name for layer in extract_layers(network) if
                not isinstance(layer, (tf.keras.layers.Activation, tf.keras.layers.InputLayer))]
    else:
        return [layer.name for layer in extract_layers(network)]


def extract_layers(network: tf.keras.Model, unfold_tds: bool = False) -> List[tf.keras.layers.Layer]:
    """Recursively extract layers from a potentially nested list of Sequentials of unknown depth."""
    layers = []
    for l in network.layers:
        if isinstance(l, tf.keras.Model) or isinstance(l, tf.keras.Sequential):
            layers.append(extract_layers(l))
        elif isinstance(l, tf.keras.layers.TimeDistributed) and unfold_tds:
            if isinstance(l.layer, tf.keras.Model) or isinstance(l.layer, tf.keras.Sequential):
                layers.append(extract_layers(l.layer))
            else:
                layers.append(l.layer)
        else:
            layers.append(l)

    return flatten(layers)


def insert_unknown_shape_dimensions(shape, none_replacer: int = 1):
    """Replace Nones in a shape tuple with 1 or a given other value."""
    return tuple(map(lambda s: none_replacer if s is None else s, shape))


def reset_states_masked(model: tf.keras.Model, mask: List):
    """Reset a stateful model's states only at the samples in the batch that are specified by the mask.

    The mask should be a list of length 'batch size' and contain one at every position where the state should be reset,
    and zeros otherwise (booleans possible too)."""

    # extract recurrent layers by their superclass RNN
    recurrent_layers = [layer for layer in extract_layers(model) if isinstance(layer, tf.keras.layers.RNN)]

    for layer in recurrent_layers:
        current_states = [state.numpy() for state in layer.states]
        initial_states = 0
        new_states = []
        for current_state in current_states:
            expanded_mask = numpy.tile(numpy.rot90(numpy.expand_dims(mask, axis=0)), (1, current_state.shape[-1]))
            masked_reset_state = np.where(expanded_mask, initial_states, current_state)
            new_states.append(masked_reset_state)

        layer.reset_states(new_states)


def detect_finished_episodes(action_log_probabilities: tf.Tensor):
    """Detect which samples in the batch connect to a episode that finished during the subsequence, based on the action
    log probabilities and return a 1D boolean tensor.

    Input Shape:
        action_probabilities: (B, S)
    """
    # TODO wont work for episodes that finish exactly at end of sequence
    # need to check only last one, as checking any might catch (albeit unlikely) true 0 in the sequence
    finished = action_log_probabilities[:, -1] == 0
    return finished


def get_layer_names(model: tf.keras.Model):
    """Get names of all (outer) layers in the model."""
    return [layer.name if not isinstance(layer, TimeDistributed) else layer.layer.name for layer in model.layers]


def get_component(model: tf.keras.Model, name: str):
    """Get outer layer/component by name."""
    for layer in model.layers:
        layer_name = layer.name
        if isinstance(layer, TimeDistributed):
            layer_name = layer.layer.name

        if layer_name == name:
            return layer


def calc_max_memory_usage(model: tf.keras.Model):
    """Calculate memory requirement of a model per sample in bits."""
    layers = extract_layers(model)
    n_shapes = int(numpy.sum(
        [numpy.prod(numpy.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in layers]))
    n_parameters = model.count_params()

    # memory needed for saving activations during gradient calculation
    n_activations = 0
    for l in layers:
        if len(l.trainable_variables) == 0 or l.output_shape is None:
            continue

        activation_shapes = l.output_shape
        if not isinstance(activation_shapes[0], tuple):
            activation_shapes = [tuple(activation_shapes)]

        print(activation_shapes)

    print(n_activations)

    total_memory = (n_shapes + n_parameters + n_activations) * 32

    return total_memory * 1.1641532182693481 * 10 ** -10


if __name__ == "__main__":
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
