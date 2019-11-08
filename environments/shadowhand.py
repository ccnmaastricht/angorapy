#!/usr/bin/env python
"""ShadowHand Environment Wrappers."""
import os

import gym
import numpy
from gym.envs.robotics import HandBlockEnv
import matplotlib.pyplot as plt


class ShadowHand(HandBlockEnv):
    """Wrapper for the In Hand Manipulation Environment."""

    def __init__(self, target_position='random', target_rotation='xyz', reward_type='not_sparse', max_steps=100):
        super().__init__(target_position, target_rotation, reward_type)

        self.max_steps = max_steps
        self.total_steps = 0

        # color of the hand
        self.sim.model.mat_rgba[2] = numpy.array([16, 18, 35, 255]) / 255

        # background color
        self.sim.model.mat_rgba[4] = numpy.array([104, 143, 71, 255]) / 255

    def _viewer_setup(self):
        super()._viewer_setup()

        # rotate camera to top down view
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = -90.0
        self.viewer.cam.elevation = -90.0

    def _convert_obs(self, obs):
        # frame = self.render(mode="rgb_array", height=200, width=200)
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


if __name__ == "__main__":
    print(os.environ["LD_PRELOAD"])

    from environments import *

    env = gym.make("ShadowHand-v0")
    done = False
    state = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
