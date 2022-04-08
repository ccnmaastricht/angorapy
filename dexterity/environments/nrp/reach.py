"""NRP implementations of reaching tasks."""
import collections
import random
from typing import Union, Callable

import mujoco_py
import numpy as np
from gym.envs.robotics.hand.reach import DEFAULT_INITIAL_QPOS, HandReachEnv, goal_distance
from gym.envs.robotics.utils import robot_get_obs
from mujoco_py import GlfwContext

from common.const import VISION_WH, N_SUBSTEPS
from common.reward import reach
from common.senses import Sensation
from configs.reward_config import resolve_config_name, REACH_BASE
from environments.nrp.shadowhand import FINGERTIP_SITE_NAMES, generate_random_sim_qpos, BaseNRPShadowHandEnv, \
    get_fingertip_distance
from utilities.util import HiddenPrints


class NRPShadowHandReachSimple(HandReachEnv):
    """Simple Implementation of the Reaching task, using NRP backend."""

    def __init__(self, distance_threshold=0.02, n_substeps=20, relative_control=True,
                 initial_qpos=DEFAULT_INITIAL_QPOS, reward_type='dense', success_multiplier=0.1):
        super().__init__(distance_threshold, n_substeps, relative_control, initial_qpos, reward_type)
        self.success_multiplier = success_multiplier

    def compute_reward(self, achieved_goal, goal, info):
        """Compute reward with additional success bonus."""
        return super().compute_reward(achieved_goal, goal, info) + info["is_success"] * self.success_multiplier

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().ravel()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal, self.goal.copy()])

        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }


class NRPShadowHandReach(BaseNRPShadowHandEnv):
    """Simple Reaching task (NRP IMPLEMENTATION)."""

    def __init__(self, initial_qpos=DEFAULT_INITIAL_QPOS, n_substeps=N_SUBSTEPS, relative_control=True, vision=False,
                 touch=True):
        if vision:
            with HiddenPrints():
                # fix to "ERROR: GLEW initalization error: Missing GL version"
                mujoco_py.GlfwContext(offscreen=True)

        # reward function setup
        self._set_default_reward_function_and_config()
        self.vision = vision
        self.touch = touch

        self.current_target_finger = 3
        self.thumb_name = 'robot0:S_thtip'

        # STATE INITIALIZATION
        assert initial_qpos in ["random", "buffered"] or isinstance(initial_qpos, dict), "Illegal state initialization."

        self.state_initialization = initial_qpos
        if self.state_initialization == "random":
            initial_qpos = generate_random_sim_qpos(DEFAULT_INITIAL_QPOS)
        elif self.state_initialization == "buffered":
            initial_qpos = DEFAULT_INITIAL_QPOS

        super().__init__(initial_qpos, self.reward_config["SUCCESS_DISTANCE"], n_substeps, relative_control)

        self.previous_finger_positions = [self.get_finger_position(fname) for fname in FINGERTIP_SITE_NAMES]

    def _set_default_reward_function_and_config(self):
        self.reward_function = reach
        self.reward_config = REACH_BASE

    def assert_reward_setup(self):
        """Assert whether the reward config fits the environment. """
        assert set(REACH_BASE.keys()).issubset(self.reward_config.keys()), "Incomplete free reach reward configuration."

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = get_fingertip_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def get_thumb_position(self) -> np.ndarray:
        """Get the position of the thumb in space."""
        return self.sim.data.get_site_xpos(self.thumb_name).flatten()

    def get_thumbs_previous_position(self) -> np.ndarray:
        """Get the previous position of the thumb in space."""
        return self.previous_finger_positions[-1]

    def get_fingers_previous_position(self, fname) -> np.ndarray:
        """Get the previous position of the given finger in space."""
        return self.previous_finger_positions[FINGERTIP_SITE_NAMES.index(fname)]

    def get_target_finger_position(self) -> np.ndarray:
        """Get position of the target finger in space."""
        return self.sim.data.get_site_xpos(FINGERTIP_SITE_NAMES[self.current_target_finger]).flatten()

    def get_target_fingers_previous_position(self) -> np.ndarray:
        """Get position of the target finger in space."""
        return self.previous_finger_positions[self.current_target_finger]

    def get_finger_position(self, finger_name: str) -> np.ndarray:
        """Get position of the specified finger in space."""
        return self.sim.data.get_site_xpos(finger_name).flatten()

    def compute_reward(self, achieved_goal, goal, info):
        """Compute reward with additional success bonus."""
        return self.reward_function(self, achieved_goal, goal, info)

    def set_reward_function(self, function: Union[str, Callable]):
        """Set the environment reward function by its config identifier or a callable."""
        if isinstance(function, str):
            try:
                function = getattr(reward, function.split(".")[0])
            except AttributeError:
                raise AttributeError("Reward function unknown.")

        self.reward_function = function

    def set_reward_config(self, new_config: Union[str, dict]):
        """Set the environment'serialization reward configuration by its identifier or a dict."""
        if isinstance(new_config, str):
            new_config: dict = resolve_config_name(new_config)

        self.reward_config = new_config
        self.distance_threshold = self.reward_config["SUCCESS_DISTANCE"]

        self.assert_reward_setup()

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful."""
        self.sim.set_state(self.initial_state)  # reset everything

        if self.state_initialization == "random":
            # generate and set random initial state
            initial_qpos = generate_random_sim_qpos(DEFAULT_INITIAL_QPOS)

            for name, value in initial_qpos.items():
                self.sim.data.set_joint_qpos(name, value)
                self.sim.data.set_joint_qvel(name, 0)
        # elif self.state_initialization == "buffered":
        #     # pull state from buffer as initial state for the environment
        #     if len(self.state_memory_buffer) > 100:  # TODO as variable in config?
        #         sampled_initial_state = random.choice(self.state_memory_buffer)
        #         self.sim.set_state(sampled_initial_state)

        self.sim.forward()
        return True

    def _get_obs(self):
        touch = self.sim.data.sensordata[self._touch_sensor_id]

        achieved_goal = self._get_achieved_goal().ravel()

        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        proprioception = np.concatenate([robot_qpos, robot_qvel, achieved_goal.copy()])  # todo remove achieved goal?

        return {
            'observation': Sensation(
                proprioception=proprioception,
                somatosensation=touch if self.touch else None,
                vision=self.render("rgb_array", VISION_WH, VISION_WH) if self.vision else None,
                goal=self.goal.copy()
            ),

            'desired_goal': self.goal.copy(),
            'achieved_goal': achieved_goal.copy(),
        }

    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = self.np_random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        self.current_target_finger = finger_idx

        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if self.np_random.uniform() < 0.1:
            goal = self.initial_goal.copy()

        return goal.flatten()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()

    def _render_callback(self, render_targets=False):
        if render_targets:
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

            # Visualize targets.
            goal = self.goal.reshape(5, 3)
            for finger_idx in range(5):
                site_name = 'target{}'.format(finger_idx)
                site_id = self.sim.model.site_name2id(site_name)
                self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]
                self.sim.model.site_rgba[site_id][-1] = 0.2

            # Visualize finger positions.
            achieved_goal = self._get_achieved_goal().reshape(5, 3)
            for finger_idx in range(5):
                site_name = 'finger{}'.format(finger_idx)
                site_id = self.sim.model.site_name2id(site_name)
                self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]
                self.sim.model.site_rgba[site_id][-1] = 0.2

        self.sim.forward()

    def step(self, action):
        """Step the environment."""
        self.previous_finger_positions = [self.get_finger_position(fname).copy() for fname in FINGERTIP_SITE_NAMES]

        o, r, d, i = super().step(action)

        # update memory
        i.update({"target_finger": self.current_target_finger})
        # self.state_memory_buffer.append(self.sim.get_state())

        return o, r, d, i