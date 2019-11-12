#!/usr/bin/env python
"""ShadowHand Environment Wrappers."""
import os

import numpy
from gym import utils, spaces
from gym.envs.robotics.hand import manipulate
from gym.envs.robotics.hand.manipulate_touch_sensors import MANIPULATE_BLOCK_XML, MANIPULATE_EGG_XML
from mujoco_py import GlfwContext


class ShadowHand(manipulate.ManipulateEnv):
    """Adjusted version of ManipulateTouchSensorsEnv Environment in the gym package to fit the projects needs."""

    def __init__(self, model_path, target_position, target_rotation, target_position_range, reward_type,
                 initial_qpos={}, randomize_initial_position=True, randomize_initial_rotation=True,
                 distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False,
                 ignore_z_target_rotation=False, touch_visualisation="on_touch", touch_get_obs="sensordata",
                 visual_input: bool = False, max_steps=100):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation
            visual_input (bool): indicator whether the environment should return frames (True) or the exact object
                position (False)
            max_steps (int): maximum number of steps before episode is ended
        """
        # init rendering [IMPORTANT]
        GlfwContext(offscreen=True)

        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]
        self.visual_input = visual_input
        self.total_steps = 0
        self.max_steps = max_steps

        manipulate.ManipulateEnv.__init__(
            self, model_path, target_position, target_rotation,
            target_position_range, reward_type, initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold, rotation_threshold=rotation_threshold, n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
        )

        # get touch sensor site names and their ids
        for k, v in self.sim.model._sensor_name2id.items():
            if 'robot0:TS_' in k:
                self._touch_sensor_id_site_id.append(
                    (v, self.sim.model._site_name2id[k.replace('robot0:TS_', 'robot0:T_')]))
                self._touch_sensor_id.append(v)

        # set touch sensors rgba values
        if self.touch_visualisation == 'off':
            for _, site_id in self._touch_sensor_id_site_id:
                self.sim.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == 'always':
            pass

        # set hand and background colors
        self.sim.model.mat_rgba[2] = numpy.array([16, 18, 35, 255]) / 255
        self.sim.model.mat_rgba[4] = numpy.array([104, 143, 71, 255]) / 255

        # set observation space
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-numpy.inf, numpy.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-numpy.inf, numpy.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Tuple((
                spaces.Box(-numpy.inf, numpy.inf, shape=obs["observation"][0].shape, dtype='float32'),  # visual/object
                spaces.Box(-numpy.inf, numpy.inf, shape=obs["observation"][1].shape, dtype='float32'),  # proprioception
                spaces.Box(-numpy.inf, numpy.inf, shape=obs["observation"][2].shape, dtype='float32'),  # touch sensors
                spaces.Box(-numpy.inf, numpy.inf, shape=obs["observation"][3].shape, dtype='float32'),  # goal
            ))
        ))

    def _viewer_setup(self):
        super()._viewer_setup()

        # rotate camera to top down view
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = -90.0
        self.viewer.cam.elevation = -90.0

    def _render_callback(self):
        super()._render_callback()
        if self.touch_visualisation == 'on_touch':
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.sim.data.sensordata[touch_sensor_id] != 0.0:
                    self.sim.model.site_rgba[site_id] = self.touch_color
                else:
                    self.sim.model.site_rgba[site_id] = self.notouch_color

    def _get_obs(self):
        # "primary" information, either this is the visual frame or the object position and velocity
        achieved_goal = self._get_achieved_goal().ravel()
        if self.visual_input:
            primary = self.render(mode="rgb_array", height=200, width=200)
        else:
            object_vel = self.sim.data.get_joint_qvel('object:joint')
            primary = numpy.concatenate(achieved_goal, object_vel)

        # get proprioceptive information (positions of joints)
        robot_pos, robot_vel = manipulate.robot_get_obs(self.sim)
        proprioception = numpy.concatenate([robot_pos, robot_vel])

        # touch sensor information
        if self.touch_get_obs == 'sensordata':
            touch = self.sim.data.sensordata[self._touch_sensor_id]
        else:
            raise NotImplementedError("Only sensor data supported atm, sorry.")

        return {
            "observation": numpy.array((primary.copy(), proprioception.copy(), touch.copy(), self.goal.ravel().copy())),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

    def step(self, action):
        """Make step in environment."""
        self.total_steps += 1
        obs, reward, done, info = super().step(action)
        return obs, reward, done if self.total_steps < self.max_steps else True, info

    def reset(self):
        """Reset the environment."""
        self.total_steps = 0
        return super().reset()


class ShadowHandBlock(ShadowHand, utils.EzPickle):
    """ShadowHand Environment with a Block as an object."""

    def __init__(self, target_position='random', target_rotation='xyz', touch_get_obs='sensordata',
                 reward_type='sparse', visual_input: bool = False, max_steps=100):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, reward_type)
        ShadowHand.__init__(self,
                            model_path=MANIPULATE_BLOCK_XML,
                            touch_get_obs=touch_get_obs,
                            target_rotation=target_rotation,
                            target_position=target_position,
                            target_position_range=numpy.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                            reward_type=reward_type,
                            visual_input=visual_input,
                            max_steps=max_steps)


class ShadowHandEgg(ShadowHand, utils.EzPickle):
    """ShadowHand Environment with an Egg as an object."""

    def __init__(self, target_position='random', target_rotation='xyz', touch_get_obs='sensordata',
                 reward_type='sparse', visual_input: bool = False, max_steps=100):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, reward_type)
        ShadowHand.__init__(self,
                            model_path=MANIPULATE_EGG_XML,
                            touch_get_obs=touch_get_obs,
                            target_rotation=target_rotation,
                            target_position=target_position,
                            target_position_range=numpy.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                            reward_type=reward_type,
                            visual_input=visual_input,
                            max_steps=max_steps)


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
