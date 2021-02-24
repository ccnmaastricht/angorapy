import os

import numpy as np
from gym import utils, spaces
from gym.envs.robotics import HandBlockEnv
from gym.envs.robotics.hand import manipulate
from gym.envs.robotics.utils import robot_get_obs

from common.senses import Sensation
from environments.shadowhand import BaseShadowHandEnv, get_palm_position
from utilities.const import VISION_WH, N_SUBSTEPS

MANIPULATE_BLOCK_XML = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))),
                                    "assets/hand",
                                    'manipulate_block_touch_sensors.xml')
MANIPULATE_EGG_XML = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))),
                                  "assets/hand",
                                  'manipulate_egg_touch_sensors.xml')


class BaseManipulate(BaseShadowHandEnv, manipulate.ManipulateEnv):
    """Base class for in-hand manipulation tasks."""

    def __init__(self, model_path, target_position, target_rotation, target_position_range, reward_type,
                 initial_qpos={}, randomize_initial_position=True, randomize_initial_rotation=True,
                 distance_threshold=0.01, rotation_threshold=0.1, n_substeps=N_SUBSTEPS, relative_control=True,
                 ignore_z_target_rotation=False, touch_visualisation="off", touch_get_obs="sensordata",
                 visual_input: bool = False, max_steps=200):
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

        if visual_input:
            # init rendering [IMPORTANT]
            from mujoco_py import GlfwContext
            GlfwContext(offscreen=True)  # in newer version of gym use quiet=True to silence this

        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self.visual_input = visual_input
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]
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

        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []

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
        self.sim.model.geom_rgba[48] = np.array([0.5, 0.5, 0.5, 0])

        # set observation space
        self.observation_space = self._determine_observation_space()

    def _determine_observation_space(self):
        obs = self._get_obs()
        return spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Tuple((
                spaces.Box(-np.inf, np.inf, shape=obs["observation"][0].shape, dtype='float32'),  # visual/object
                spaces.Box(-np.inf, np.inf, shape=obs["observation"][1].shape, dtype='float32'),  # proprioception
                spaces.Box(-np.inf, np.inf, shape=obs["observation"][2].shape, dtype='float32'),  # touch sensors
                spaces.Box(-np.inf, np.inf, shape=obs["observation"][3].shape, dtype='float32'),  # goal
            ))
        ))

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
            primary = self.render(mode="rgb_array", height=VISION_WH, width=VISION_WH)
        else:
            object_vel = self.sim.data.get_joint_qvel('object:joint')
            primary = np.concatenate([achieved_goal, object_vel])

        # get proprioceptive information (positions of joints)
        robot_pos, robot_vel = manipulate.robot_get_obs(self.sim)
        proprioception = np.concatenate([robot_pos, robot_vel])

        # touch sensor information
        if self.touch_get_obs == 'sensordata':
            touch = self.sim.data.sensordata[self._touch_sensor_id]
        else:
            raise NotImplementedError("Only sensor data supported atm, sorry.")

        return {
            "observation": Sensation(
                vision=primary.copy(),
                proprioception=proprioception.copy(),
                somatosensation=touch.copy(),
                goal=self.goal.ravel().copy()),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

    def _is_success(self, achieved_goal, desired_goal):
        """We determine success only by means of rotational goals."""
        _, d_rot = self._goal_distance(achieved_goal, desired_goal)
        return (d_rot < self.rotation_threshold).astype(np.float32)

    def _is_dropped(self) -> bool:
        """Heuristically determine whether the object still is in the hand."""

        # determin object center position
        obj_center_idx = self.sim.model.site_name2id('object:center')
        obj_center_pos = self.sim.data.site_xpos[obj_center_idx]

        # determine palm center position
        palm_center_pos = get_palm_position(self.sim)

        dropped = (
                obj_center_pos[2] < palm_center_pos[2]  # z axis of object smaller than that of palm
            # we could add smth like checking for contacts between palm and object here, but the above works
            # pretty well already tbh
        )

        return dropped

    def step(self, action):
        """Make step in environment."""
        self.total_steps += 1
        obs, reward, done, info = super().step(action)
        dropped = self._is_dropped()
        done = done or dropped or self.total_steps >= self.max_steps

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        """Compute the reward."""
        success = self._is_success(achieved_goal, goal).astype(np.float32)
        _, d_rot = self._goal_distance(achieved_goal, goal)

        return (- d_rot  # convergence to goal reward
                - 1  # constant punishment to encourage fast solutions
                + success * 5  # reward for finishing
                + 20 * self._is_dropped())  # dropping penalty

    def reset(self):
        """Reset the environment."""
        self.total_steps = 0
        return super().reset()


class ManipulateBlock(BaseManipulate, utils.EzPickle):
    """Manipulate Environment with a Block as an object."""

    def __init__(self, target_position='ignore', target_rotation='xyz', touch_get_obs='sensordata',
                 reward_type='dense', visual_input: bool = False, max_steps=200):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, reward_type)
        BaseManipulate.__init__(self,
                                model_path=MANIPULATE_BLOCK_XML,
                                touch_get_obs=touch_get_obs,
                                target_rotation=target_rotation,
                                target_position=target_position,
                                target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                                reward_type=reward_type,
                                visual_input=visual_input,
                                max_steps=max_steps)


class ManipulateEgg(BaseManipulate, utils.EzPickle):
    """Manipulate Environment with an Egg as an object."""

    def __init__(self, target_position='ignore', target_rotation='xyz', touch_get_obs='sensordata',
                 reward_type='dense', visual_input: bool = False, max_steps=200):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, reward_type)
        BaseManipulate.__init__(self,
                                model_path=MANIPULATE_EGG_XML,
                                touch_get_obs=touch_get_obs,
                                target_rotation=target_rotation,
                                target_position=target_position,
                                target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                                reward_type=reward_type,
                                visual_input=visual_input,
                                max_steps=max_steps)


class ManipulateBlockVector(HandBlockEnv):

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal, self.goal.ravel().copy()])

        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }
