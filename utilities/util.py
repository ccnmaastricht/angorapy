#!/usr/bin/env python
"""Helper functions."""
import random
from typing import Tuple

import gym
from gym.spaces import Discrete, Box
import tensorflow as tf
import numpy as np


def flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    print(f"\r{string}", end="")


def set_all_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def env_extract_dims(env: gym.Env) -> Tuple[int, int]:
    """Returns state and (discrete) action space dimensionality for given environment."""
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
    elif isinstance(env.action_space, Box):
        act_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Environment has unknown Action Space Typ: {env.action_space}")

    return obs_dim, act_dim


def normalize(x, is_img=False):
    """Normalize a numpy array to have all values in range (0, 1)."""
    x = tf.convert_to_tensor(x).numpy()
    if not is_img:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return x / 255


def resize_normalize(d: tf.data.Dataset):
    return d.map(lambda img: (tf.image.resize(img["image"], (200, 200)) / 255,) * 2)


def flatten(l):
    return [l] if not isinstance(l, list) else [x for X in l for x in flatten(X)]
