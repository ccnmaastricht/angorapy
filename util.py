#!/usr/bin/env python
"""Helper functions."""
from typing import Tuple

import gym


def flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    print(f"\r{string}", end="")


def env_extract_dims(env: gym.Env) -> Tuple[int, int]:
    """Returns state and (discrete) action space dimensionality for given environment."""
    return env.observation_space.shape[0], env.action_space.n
