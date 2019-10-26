#!/usr/bin/env python
"""Helper functions."""
from typing import Tuple

import gym
from gym.spaces import Discrete, Box


def flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    print(f"\r{string}", end="")


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


def flatten(l):
    return [l] if not isinstance(l, list) else [x for X in l for x in flatten(X)]