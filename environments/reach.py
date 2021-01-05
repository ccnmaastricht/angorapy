import random
from typing import Union, Callable

import numpy as np
from gym import spaces
from gym.envs.robotics import HandReachEnv
from gym.envs.robotics.hand import manipulate
from gym.envs.robotics.hand.reach import DEFAULT_INITIAL_QPOS, FINGERTIP_SITE_NAMES
from gym.envs.robotics.utils import robot_get_obs

from configs.reward_config import REACH_BASE, resolve_config_name
from core import reward
from core.reward import sequential_free_reach, free_reach, reach
from environments.shadowhand import get_fingertip_distance, generate_random_sim_qpos, BaseShadowHand
from utilities.const import N_SUBSTEPS, VISION_WH


class Reach(HandReachEnv, BaseShadowHand):
    """Simple Reaching task."""

    def assert_reward_setup(self):
        """Assert whether the reward config fits the environment. """
        assert set(REACH_BASE.keys()).issubset(self.reward_config.keys()), "Incomplete free reach reward configuration."

    def __init__(self, n_substeps=N_SUBSTEPS, relative_control=True, initial_qpos=DEFAULT_INITIAL_QPOS):
        # reward function setup
        self.reward_function = reach
        self.reward_config = REACH_BASE

        self.current_target_finger = None
        self.thumb_name = 'robot0:S_thtip'

        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []

        self.random_initial_state = initial_qpos == "random"
        if self.random_initial_state:
            initial_qpos = generate_random_sim_qpos(DEFAULT_INITIAL_QPOS)

        super().__init__(self.reward_config["SUCCESS_DISTANCE"], n_substeps, relative_control, initial_qpos, "dense")

        for k, v in self.sim.model._sensor_name2id.items():
            if 'robot0:TS_' in k:
                self._touch_sensor_id_site_id.append(
                    (v, self.sim.model._site_name2id[k.replace('robot0:TS_', 'robot0:T_')]))
                self._touch_sensor_id.append(v)

        self.previous_finger_positions = [self.get_finger_position(fname) for fname in FINGERTIP_SITE_NAMES]

    def get_thumb_position(self) -> np.ndarray:
        """Get the position of the thumb in space."""
        return self.sim.data.get_site_xpos(self.thumb_name).flatten()

    def get_thumbs_previous_position(self) -> np.ndarray:
        """Get the position of the thumb in space."""
        return self.previous_finger_positions[-1]

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
        self.reward_function(self, info, achieved_goal, goal)

    def set_reward_function(self, function: Union[str, Callable]):
        """Set the environment reward function by its config identifier or a callable."""
        if isinstance(function, str):
            try:
                function = getattr(reward, function.split(".")[0])
            except AttributeError:
                raise AttributeError("Reward function unknown.")

        self.reward_function = function

    def set_reward_config(self, new_config: Union[str, dict]):
        """Set the environment's reward configuration by its identifier or a dict."""
        if isinstance(new_config, str):
            new_config: dict = resolve_config_name(new_config)

        self.reward_config = new_config
        self.distance_threshold = self.reward_config["SUCCESS_DISTANCE"]

        self.assert_reward_setup()

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful."""
        self.sim.set_state(self.initial_state)  # reset everything

        # if wanted, generate and set random initial state
        if self.random_initial_state:
            initial_qpos = generate_random_sim_qpos(DEFAULT_INITIAL_QPOS)

            for name, value in initial_qpos.items():
                self.sim.data.set_joint_qpos(name, value)
                self.sim.data.set_joint_qvel(name, 0)

        self.sim.forward()
        return True

    def _get_obs(self):
        # proprioception
        robot_qpos, robot_qvel = robot_get_obs(self.sim)

        # touch sensor information
        touch = self.sim.data.sensordata[self._touch_sensor_id]

        achieved_goal = self._get_achieved_goal().ravel()
        observation = np.concatenate([robot_qpos, robot_qvel, touch, achieved_goal, self.goal.copy()])

        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
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

    def step(self, action):
        """Step the environment."""
        self.previous_finger_positions = [self.get_finger_position(fname).copy() for fname in FINGERTIP_SITE_NAMES]

        o, r, d, i = super().step(action)
        i.update({"target_finger": self.current_target_finger})

        return o, r, d, i


class MultiReach(Reach):
    """Reaching task where three fingers have to be joined."""

    def __init__(self, n_substeps=N_SUBSTEPS, relative_control=True,
                 initial_qpos=DEFAULT_INITIAL_QPOS):
        super().__init__(n_substeps, relative_control, initial_qpos)

    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]

        # choose the fingers to join with the thumb
        finger_name_a, finger_name_b = self.np_random.choice(a=finger_names, size=2, replace=False)

        # retrieve their indices
        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx_a = FINGERTIP_SITE_NAMES.index(finger_name_a)
        finger_idx_b = FINGERTIP_SITE_NAMES.index(finger_name_b)

        # pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx_a, finger_idx_b]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.007 * offset_direction

        if self.np_random.uniform() < 0.1:
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()

        return goal.flatten()


class FreeReach(Reach):
    """Reaching task where the actual position of the joint fingers is irrelevant.

    It only matters which fingertips need to be joint. The reward is based on the distance between the fingertips,
    punishing distance of the thumb to target fingers and rewarding the distance to non-target fingers.

    The goal is represented as a one-hot vector of size 4."""

    def __init__(self, n_substeps=N_SUBSTEPS, relative_control=True, initial_qpos=DEFAULT_INITIAL_QPOS,
                 force_finger=None):
        assert force_finger in list(range(5)) + [None], "Forced finger index out of range [0, 5]."

        self.forced_finger = force_finger
        super().__init__(n_substeps, relative_control, initial_qpos)

        self.reward_function = free_reach

    def compute_reward(self, achieved_goal, goal, info):
        return self.reward_function(self, achieved_goal, goal, info)

    def _sample_goal(self):
        if self.forced_finger is None:
            finger_names = [name for name in FINGERTIP_SITE_NAMES if name != self.thumb_name]

            # choose the finger to join with the thumb
            finger_name = self.np_random.choice(a=finger_names, size=1, replace=False)

            # get finger id
            f_id = FINGERTIP_SITE_NAMES.index(finger_name)
        else:
            f_id = self.forced_finger

        self.current_target_finger = f_id

        # make one hot encoding
        goal = np.zeros(len(FINGERTIP_SITE_NAMES))
        goal[f_id] = 1

        return goal

    def get_target_finger_position(self):
        """Get position of the target finger in space."""
        return self.sim.data.get_site_xpos(FINGERTIP_SITE_NAMES[np.where(self.goal == 1)[0].item()]).flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = get_fingertip_distance(self.get_thumb_position(), self.get_target_finger_position())
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)

            fname = FINGERTIP_SITE_NAMES[finger_idx]
            if not (fname == self.thumb_name or finger_idx == np.where(self.goal == 1)[0].item()):
                self.sim.model.site_rgba[site_id][-1] = 0
                continue

            self.sim.model.site_rgba[site_id][-1] = 0.2
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]

        self.sim.forward()


class FreeReachSequential(FreeReach):
    """Freely join fingers in a sequence, where each new goal is randomly generated as the old goal is reached."""

    def __init__(self, n_substeps=N_SUBSTEPS, relative_control=True, initial_qpos=DEFAULT_INITIAL_QPOS):
        self.current_target_finger = None
        self.goal_sequence = []

        super().__init__(n_substeps, relative_control, initial_qpos)

        self.reward_function = sequential_free_reach
        self.reward_config = REACH_BASE

    def _sample_goal(self):
        available_fingers = [0, 1, 2, 3]
        available_fingers.remove(self.current_target_finger)
        self.current_target_finger = random.choice(available_fingers)

        # make one hot encoding
        goal = np.zeros(len(FINGERTIP_SITE_NAMES))
        goal[self.current_target_finger] = 1

        self.goal_sequence.append(self.current_target_finger)

        return goal

    def get_target_finger_position(self) -> np.ndarray:
        return self.sim.data.get_site_xpos(FINGERTIP_SITE_NAMES[self.current_target_finger]).flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = get_fingertip_distance(self.get_thumb_position(),
                                   self.get_finger_position(FINGERTIP_SITE_NAMES[self.current_target_finger]))
        return (d < self.distance_threshold).astype(np.float32)

    def reset(self):
        ret = super().reset()
        self.goal_sequence = []

        return ret

    def _render_callback(self):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            fname = FINGERTIP_SITE_NAMES[finger_idx]
            site_name = 'finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            if not (fname == self.thumb_name or finger_idx == self.current_target_finger):
                self.sim.model.site_rgba[site_id][-1] = 0
                continue

            self.sim.model.site_rgba[site_id][-1] = 0.2
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]

        self.sim.forward()

    def step(self, action):
        observation, reward, done, info = super().step(action)

        # set next subgoal if current one is achieved
        if info["is_success"]:
            print(f"Reached Target {self.current_target_finger} (after reaching {len(self.goal_sequence) - 1} targets)!")
            self.goal = self._sample_goal()

        return observation, reward, done, info