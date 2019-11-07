#!/usr/bin/env python
"""ShadowHand Environment Wrappers."""
import numpy
from gym import utils
from gym.envs.robotics import HandBlockEnv
import matplotlib.pyplot as plt

class ShadowHand(HandBlockEnv):
    """Wrapper for the In Hand Manipulation Environment."""

    def __init__(self, target_position='random', target_rotation='xyz', reward_type='not_sparse', max_steps=100):
        super().__init__(target_position, target_rotation, reward_type)

        self.max_steps = max_steps
        self.total_steps = 0

    def _viewer_setup(self):
        super()._viewer_setup()

        # rotate camera to top down view
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = -90.0
        self.viewer.cam.elevation = -90.0

    def _convert_obs(self, obs):
        frame = self.render(mode="rgb_array", height=200, width=200)
        return numpy.concatenate([obs["observation"], obs["desired_goal"]])

    def step(self, action):
        self.total_steps += 1
        obs, reward, done, info = super().step(action)
        if self.total_steps >= self.max_steps:
            done = True

        return self._convert_obs(obs), reward, done, info

    def reset(self):
        obs = super().reset()
        self.total_steps = 0
        return self._convert_obs(obs)
