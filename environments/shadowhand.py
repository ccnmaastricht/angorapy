#!/usr/bin/env python
"""BaseShadowHand Environment Wrappers."""
import abc
import random

import numpy as np
from gym.envs.robotics.hand import manipulate


def generate_random_sim_qpos(base: dict) -> dict:
    """Generate a random state of the simulation."""
    for key, val in base.items():
        if key in ["robot0:WRJ1", "robot0:WRJ0"]:
            continue  # do not randomize the wrist

        base[key] = val + val * random.gauss(0, 0.1)

    return base


def get_palm_position(sim):
    """Return the robotic hand'serialization palm'serialization center."""
    palm_idx = sim.model.body_names.index("robot0:palm")
    return np.array(sim.model.body_pos[palm_idx])


def get_fingertip_distance(ft_a, ft_b):
    """Return the distance between two vectors representing finger tip positions."""
    assert ft_a.shape == ft_b.shape
    return np.linalg.norm(ft_a - ft_b, axis=-1)


class BaseShadowHand(manipulate.ManipulateEnv, abc.ABC):
    """Base class for all shadow hand environments, setting up mostly visual characteristics of the environment."""

    def _viewer_setup(self):
        super()._viewer_setup()

        # set colors
        self.sim.model.mat_rgba[2] = np.array([16, 18, 35, 255]) / 255  # hand
        # self.sim.model.mat_rgba[2] = np.array([200, 200, 200, 255]) / 255  # hand
        self.sim.model.mat_rgba[4] = np.array([71, 116, 144, 255]) / 255  # background
        # self.sim.model.geom_rgba[48] = np.array([0.5, 0.5, 0.5, 0])

        self.viewpoint = "topdown"

        if self.viewpoint == "topdown":
            # rotate camera to top down view
            self.viewer.cam.distance = 0.35  # zoom in
            self.viewer.cam.azimuth = -0.0  # top down view
            self.viewer.cam.elevation = -90.0  # top down view
            self.viewer.cam.lookat[1] -= 0.07  # slightly move forward
        elif self.viewpoint == "side":
            # rotate camera to side view
            self.viewer.cam.distance = 0.35  # zoom in
            self.viewer.cam.azimuth = 25.0  # top down view
            self.viewer.cam.elevation = -45.0  # top down view
            self.viewer.cam.lookat[1] -= 0.04  # slightly move forward
        else:
            raise NotImplementedError("Unknown Viewpoint.")


if __name__ == "__main__":
    from environments import *

    # env = gym.make("HandTappingAbsolute-v1")
    # env = gym.make("HandFreeReachLFAbsolute-v0")
    # env = gym.make("BaseShadowHand-v0")
    # env = gym.make("HandManipulateBlock-v0")
    env = gym.make("HandReachDenseRelative-v0")
    d, s = False, env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        s, r, d, i = env.step(action)
        if d:
            env.reset()
