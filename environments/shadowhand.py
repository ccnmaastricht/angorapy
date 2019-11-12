#!/usr/bin/env python
"""ShadowHand Environment Wrappers."""
import os

import numpy
from gym.envs.robotics import HandBlockTouchSensorsEnv
from mujoco_py import GlfwContext


class ShadowHandBase(HandBlockTouchSensorsEnv):
    """Wrapper for the In Hand Manipulation Environment."""

    def __init__(self, target_position='random', target_rotation='xyz', touch_get_obs='sensordata',
                 reward_type='not_sparse', max_steps=100):

        super().__init__(target_position=target_position, target_rotation=target_rotation,
                         touch_get_obs=touch_get_obs, reward_type=reward_type)
        self.max_steps = max_steps
        self.total_steps = 0

        # color of the hand
        self.sim.model.mat_rgba[2] = numpy.array([16, 18, 35, 255]) / 255

        # background color
        self.sim.model.mat_rgba[4] = numpy.array([104, 143, 71, 255]) / 255

        # other
        self.goal_dim = 4 + 3
        self.joint_dim = 24 + 24
        self.touch_dim = 92
        self.object_dim = 6

    def _viewer_setup(self):
        super()._viewer_setup()

        # rotate camera to top down view
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = -90.0
        self.viewer.cam.elevation = -90.0

    def _convert_obs(self, obs):
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


class ShadowHandV1(ShadowHandBase):

    def __init__(self):
        super().__init__()

        # init rendering [IMPORTANT]
        GlfwContext(offscreen=True)
        print([f.shape for f in self.reset()])

    def _convert_obs(self, obs):
        # obs is [joints, object velocity, touch sensors, object_position]

        visual = self.render(mode="rgb_array", height=200, width=200)
        proprio = obs["observation"][:self.joint_dim]
        somato = obs["observation"][self.joint_dim + self.object_dim:self.joint_dim + self.object_dim + self.touch_dim]
        goal = obs["desired_goal"]

        return visual, proprio, somato, goal


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
