"""Helper functions."""
import os
import random
import re
import sys
import threading
from contextlib import contextmanager
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import gymnasium as gym
import numpy
import numpy as np
import tensorflow as tf
from gymnasium import spaces
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gymnasium.spaces import MultiDiscrete

try:
    from mpi4py import MPI
except:
    MPI = None

from angorapy.utilities.error import UninterpretableObservationSpace


_TF_LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} [\d:.]+: [IWEF]")


def _filter_type_inference_warning(lines):
    """Yield every line except those belonging to a "Type inference failed" block.

    A block starts at the ``type_inference.cc ... Type inference failed`` line and
    runs until its terminating ``while inferring type of node ...`` line (or until
    an unrelated new TF log entry begins, whichever comes first). Pure and stateful
    so it can be unit-tested without any file-descriptor plumbing.
    """
    suppressing = False
    for line in lines:
        if suppressing:
            # A new, unrelated log entry ends the suppressed block (and is kept).
            if _TF_LOG_LINE.match(line):
                suppressing = False
                yield line
                continue
            # "while inferring type of node ..." is the block's last line.
            if "while inferring type of node" in line:
                suppressing = False
            continue
        if "Type inference failed" in line and "type_inference.cc" in line:
            suppressing = True
            continue
        yield line


@contextmanager
def suppress_type_inference_warning():
    """Suppress TensorFlow's benign "Type inference failed" grappler warning.

    When optimizing a recurrent train step on GPU, TF's type-inference pass chokes
    on the gradient of the RNN's internal control flow (a ``tf.cond`` wrapping a
    ``while_loop``): the loop's int32 time index and the float32 state/grad
    accumulators land at the same ``TFT_OPTIONAL`` tuple position, which it reports
    as incompatible. The warning is non-fatal -- grappler simply skips that one
    optimization and the graph executes correctly -- but it is emitted straight to
    the C-level stderr (fd 2), so Python/absl logging filters cannot catch it.

    This redirects only fd 2 through a filtering pump thread that drops the warning
    block, while pointing ``sys.stderr`` (used by tqdm) at the original fd so
    progress bars stay live and unfiltered.
    """
    # Only the C-level fd needs filtering; if stderr is not a real fd (e.g. a
    # captured StringIO under pytest), there is nothing to redirect -- no-op.
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return

    saved_stderr_fd = os.dup(stderr_fd)
    pipe_read_fd, pipe_write_fd = os.pipe()
    original_sys_stderr = sys.stderr

    def _pump(read_fd, out_fd):
        with os.fdopen(read_fd, "r", errors="replace") as reader, \
                os.fdopen(os.dup(out_fd), "w", errors="replace") as out:
            for line in _filter_type_inference_warning(reader):
                out.write(line)
                out.flush()

    pump = threading.Thread(target=_pump, args=(pipe_read_fd, saved_stderr_fd), daemon=True)
    pump.start()
    try:
        os.dup2(pipe_write_fd, stderr_fd)  # C-level stderr -> filter pipe
        os.close(pipe_write_fd)
        sys.stderr = os.fdopen(os.dup(saved_stderr_fd), "w")  # tqdm -> original stderr
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, stderr_fd)  # restore C-level stderr (closes pipe write end)
        pump.join(timeout=5)
        os.close(saved_stderr_fd)
        sys.stderr = original_sys_stderr


def mpi_flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\r{string}", end="")


def mpi_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
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
            obs_dim = {"proprioception": env.observation_space["observation"].shape,
                       "goal": env.observation_space["desired_goal"].shape}
        elif isinstance(env.observation_space["observation"], spaces.Dict):
            # made for sensation
            obs_dim = {name: field.shape for name, field in env.observation_space["observation"].spaces.items()}
        else:
            raise UninterpretableObservationSpace(
                f"Cannot extract the dimensionality from a spaces.Dict observation space "
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
        act_dim = (int(env.action_space.n),)
    elif isinstance(env.action_space, MultiDiscrete):
        assert np.all(env.action_space.nvec == env.action_space.nvec[0]), "Can only handle multi-discrete action" \
                                                                              "spaces where all actions have the same " \
                                                                              "number of categories."
        act_dim = (int(env.action_space.shape[0]), int(env.action_space.nvec[0].item()))
    elif isinstance(env.action_space, Box):
        act_dim = (int(env.action_space.shape[0]), 1)
    else:
        raise NotImplementedError(f"Environment has unknown Action Space Typ: {env.action_space}")

    return obs_dim, act_dim


@tf.function
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
    """For a given shape of a matrix (floor), find the shape of tiles that fit the floor and contain
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
