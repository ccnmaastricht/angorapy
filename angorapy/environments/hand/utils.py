import random

import numpy as np


def generate_random_sim_qpos(base: dict) -> dict:
    """Generate a random state of the simulation."""
    for key, val in base.items():
        if key in ["robot/WRJ1", "robot/WRJ0"]:
            continue  # do not randomize the wrist

        base[key] = val + val * random.gauss(0, 0.1)

    return base


def get_fingertip_distance(ft_a, ft_b):
    """Return the distance between two vectors representing finger tip positions."""
    assert ft_a.shape == ft_b.shape
    return np.linalg.norm(ft_a - ft_b, axis=-1)