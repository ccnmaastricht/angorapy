import random
from typing import Optional

import mujoco
import numpy as np

from angorapy.common.const import N_SUBSTEPS
from angorapy.tasks.envs.dexterity.mujoco_model.worlds.reach import ShadowHandReachWorld
from angorapy.tasks.reward_config import REACH_BASE
from angorapy.tasks.envs.dexterity.consts import DEFAULT_INITIAL_QPOS, \
    FINGERTIP_SITE_NAMES
from angorapy.tasks.envs.dexterity.core import BaseShadowHandEnv
from angorapy.tasks.envs.dexterity.utils import generate_random_sim_qpos, \
    get_fingertip_distance
from angorapy.tasks.reward import free_reach, \
    reach, \
    sequential_free_reach, \
    sequential_reach
from angorapy.tasks.utils import mj_qpos_dict_to_qpos_vector


class Reach(BaseShadowHandEnv):
    """Simple Reaching task."""

    def __init__(self,
                 initial_qpos=DEFAULT_INITIAL_QPOS,
                 n_substeps=N_SUBSTEPS,
                 relative_control=True,
                 vision=False,
                 touch=False,
                 force_finger=None,
                 render_mode: Optional[str] = None):
        assert force_finger in list(range(5)) + [None], "Forced finger index out of range [0, 5]."

        self.forced_finger = force_finger
        self.current_target_finger = self.forced_finger

        # STATE INITIALIZATION
        assert initial_qpos in ["random", "buffered"] or isinstance(initial_qpos, dict), "Illegal state initialization."

        self.state_initialization = initial_qpos
        if self.state_initialization == "random":
            initial_qpos = generate_random_sim_qpos(DEFAULT_INITIAL_QPOS)
        elif self.state_initialization == "buffered":
            initial_qpos = DEFAULT_INITIAL_QPOS

        self.world = ShadowHandReachWorld()

        super().__init__(
            initial_qpos=initial_qpos,
            n_substeps=n_substeps,
            relative_control=relative_control,
            vision=vision,
            touch=touch,
            render_mode=render_mode,
            model=self.world.root,
        )

        self.previous_finger_positions = [self.get_finger_position(fname) for fname in FINGERTIP_SITE_NAMES]

    def _set_default_reward_function_and_config(self):
        self.reward_function = reach
        self.reward_config = REACH_BASE

    def assert_reward_setup(self):
        """Assert whether the reward config fits the environment. """
        assert set(REACH_BASE.keys()).issubset(self.reward_config.keys()), "Incomplete free reach reward configuration."

    def _get_achieved_goal(self):
        return self.get_fingertip_positions()

    def _is_success(self,
                    achieved_goal,
                    desired_goal):
        d = get_fingertip_distance(achieved_goal, desired_goal)
        return (d < self.reward_config["SUCCESS_DISTANCE"]).astype(np.float32)

    def get_thumbs_previous_position(self) -> np.ndarray:
        """Get the previous position of the thumb in space."""
        return self.previous_finger_positions[-1]

    def get_fingers_previous_position(self,
                                      fname) -> np.ndarray:
        """Get the previous position of the given finger in space."""
        return self.previous_finger_positions[FINGERTIP_SITE_NAMES.index(fname)]

    def get_target_finger_position(self) -> np.ndarray:
        """Get position of the target finger in space."""
        return self.data.site(FINGERTIP_SITE_NAMES[self.current_target_finger]).xpos.flatten()

    def get_target_fingers_previous_position(self) -> np.ndarray:
        """Get position of the target finger in space."""
        return self.previous_finger_positions[self.current_target_finger]

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful."""
        self.set_state(**self.initial_state)  # reset everything

        if self.state_initialization == "random":
            initial_qpos = generate_random_sim_qpos(DEFAULT_INITIAL_QPOS)

            self.set_state(
                qpos=mj_qpos_dict_to_qpos_vector(self.model, initial_qpos),
                qvel=self.initial_state["qvel"]
            )
        # elif self.state_initialization == "buffered":
        #     # pull state from buffer as initial state for the environment
        #     if len(self.state_memory_buffer) > 100:  # TODO as variable in config?
        #         sampled_initial_state = random.choice(self.state_memory_buffer)
        #         self.sim.set_state(sampled_initial_state)

        return True

    def _sample_goal(self):
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != self.thumb_name]

        if self.forced_finger is None:
            finger_name = self.np_random.choice(finger_names)
        else:
            finger_name = finger_names[self.forced_finger]

        thumb_idx = FINGERTIP_SITE_NAMES.index(self.thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        self.current_target_finger = finger_idx
        goal = self.initial_goal.copy().reshape(-1, 3)

        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.08, 0.01, 0.05])

        # move meeting point along y-axis towards the target finger to make goal achievable
        meeting_pos[1] -= .5 * (meeting_pos - goal[finger_idx])[1]

        # perturb meeting point
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # slightly move meeting goal towards the respective finger to avoid that they overlap
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        # in 10% of the cases, the goal is the initial goal (i.e. the hand is not supposed to move)
        if self.forced_finger is None:
            if self.np_random.uniform() < 0.1:
                goal = self.initial_goal.copy()

        return goal.flatten()

    def _env_setup(self, initial_state):
        super()._env_setup(initial_state=initial_state)

        if initial_state is not None:
            self.set_state(**initial_state)

        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.data.body(self.palm_name).xpos.copy()

        self.goal = self._sample_goal()

    def _render_callback(self, render_targets=True):
        if render_targets:
            # Visualize targets.
            sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
            goal = self.goal.reshape(5, 3)
            for finger_idx in range(5):
                site_name = f"robot/target{finger_idx}"

                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                self.model.site(site_name).rgba[-1] = 0.2
                # self.data.site(site_name).xpos[:] = (goal[finger_idx] - sites_offset[site_id]).copy()
                self.model.site_pos[site_id] = (goal[finger_idx] - sites_offset[site_id]).copy()

            # Visualize finger positions.
            achieved_goal = self._get_achieved_goal().reshape(5, 3)
            for finger_idx in range(5):
                site_name = f"robot/finger{finger_idx}"
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                self.model.site(site_name).rgba[-1] = 0.2
                # self.data.site(site_name).xpos[:] = (achieved_goal[finger_idx] - sites_offset[site_id]).copy()
                self.model.site_pos[site_id] = (achieved_goal[finger_idx] - sites_offset[site_id]).copy()

        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        """Step the environment."""
        self.previous_finger_positions = [self.get_finger_position(fname).copy() for fname in FINGERTIP_SITE_NAMES]

        return super().step(action)

    def _get_info(self):
        return {
            **super()._get_info(),
            "target_finger": self.current_target_finger,
            "achieved_goal": self._get_achieved_goal().copy(),
            "desired_goal" : self.goal.copy(),
            "is_success"   : self._is_success(self._get_achieved_goal(), self.goal),
        }

    def _get_obs(self):
        """Return the observation."""
        obs = super()._get_obs()
        obs["observation"]["vision"] = self.get_fingertip_positions().flatten()

        return obs


class ReachSequential(Reach):
    """Generate finger configurations in a sequence. Each new goal is randomly generated as the old goal is reached."""

    def __init__(self,
                 initial_qpos=DEFAULT_INITIAL_QPOS,
                 n_substeps=N_SUBSTEPS,
                 relative_control=True,
                 vision=False,
                 touch=True):
        self.current_target_finger = None
        self.goal_sequence = []

        super().__init__(
            initial_qpos=initial_qpos,
            n_substeps=n_substeps,
            relative_control=relative_control,
            vision=vision,
            touch=touch)

    def _set_default_reward_function_and_config(self):
        self.reward_function = sequential_reach
        self.reward_config = REACH_BASE

    def _sample_goal(self):
        available_fingers = [0, 1, 2, 3]  # where None refers to not meeting any fingers
        if self.current_target_finger is not None:
            available_fingers.remove(self.current_target_finger)
        self.current_target_finger = random.choice(available_fingers)

        goal = self.initial_goal.copy()

        # pick a meeting point above the hand
        meeting_pos = self.palm_xpos + np.array([0.0, -0.2, 0.05])
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # slightly move meeting goal towards the respective finger to avoid that they overlap
        if self.current_target_finger is not None:
            goal = goal.reshape(-1, 3)
            for idx in [self.current_target_finger, -1]:
                offset_direction = (meeting_pos - goal[idx])
                offset_direction /= np.linalg.norm(offset_direction)
                goal[idx] = meeting_pos - 0.005 * offset_direction

        # remember the goals
        self.goal_sequence.append(self.current_target_finger)

        return goal.flatten()

    def get_target_finger_position(self) -> np.ndarray:
        return self.sim.data.get_site_xpos(FINGERTIP_SITE_NAMES[self.current_target_finger]).flatten()

    def _is_success(self,
                    achieved_goal,
                    desired_goal):
        d = get_fingertip_distance(achieved_goal, desired_goal)
        return (d < self.reward_config["SUCCESS_DISTANCE"]).astype(np.float32)

    def reset(self):
        self.current_target_finger = None
        self.goal_sequence = []

        ret = super().reset()

        return ret, {}

    def step(self,
             action):
        observation, reward, done, truncated, info = super().step(action)

        # set next subgoal if current one is achieved
        if info["is_success"]:
            # print(
            #     f"Reached Target {self.current_target_finger} (after reaching {len(self.goal_sequence) - 1}
            #     targets)!")
            self.goal = self._sample_goal()

        return observation, reward, done, truncated, info


class FreeReach(Reach):
    """Reaching task where the actual position of the joint fingers is irrelevant.

    It only matters which fingertips need to be joint. The reward is based on the distance between the fingertips,
    punishing distance of the thumb to target fingers and rewarding the distance to non-target fingers.

    The goal is represented as a one-hot vector of size 4."""

    def _set_default_reward_function_and_config(self):
        self.reward_function = free_reach
        self.reward_config = REACH_BASE

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
        goal[self.current_target_finger] = 1

        return goal

    def get_target_finger_position(self):
        """Get position of the target finger in space."""
        return self.data.site(FINGERTIP_SITE_NAMES[np.where(self.goal == 1)[0].item()]).xpos.flatten()

    def _is_success(self,
                    achieved_goal,
                    desired_goal):
        d = get_fingertip_distance(self.get_thumb_position(), self.get_target_finger_position())
        return (d < self.reward_config["SUCCESS_DISTANCE"]).astype(np.float32)

    def _render_callback(self,
                         render_targets=False):
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            site_name = f"robot/finger{finger_idx}"
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

            fname = FINGERTIP_SITE_NAMES[finger_idx]
            if not (fname == self.thumb_name or finger_idx == np.where(self.goal == 1)[0].item()):
                self.model.site_rgba[site_id][-1] = 0
                continue

            self.model.site(site_name).rgba[-1] = 0.2

            self.model.site_pos[site_id] = (
                    achieved_goal[finger_idx] - sites_offset[site_id]
            )
        mujoco.mj_forward(self.model, self.data)


class FreeReachSequential(FreeReach):
    """Freely join fingers in a sequence, where each new goal is randomly generated as the old goal is reached."""

    def __init__(self,
                 n_substeps=N_SUBSTEPS,
                 relative_control=True,
                 initial_qpos=DEFAULT_INITIAL_QPOS,
                 vision=False):
        self.goal_sequence = []

        self.force_exemplary_sequence = True
        self.default_exemplary_sequence = [0, 1, 2, 3]
        self.exemplary_sequence = self.default_exemplary_sequence.copy()

        super().__init__(n_substeps, relative_control, initial_qpos, vision)
        self.exemplary_sequence = self.default_exemplary_sequence.copy()  # reset after initial init stuff

    def _set_default_reward_function_and_config(self):
        self.reward_function = sequential_free_reach
        self.reward_config = REACH_BASE

    def _sample_goal(self):
        if not self.force_exemplary_sequence:
            available_fingers = [0, 1, 2, 3]
            available_fingers.remove(self.current_target_finger)
            self.current_target_finger = random.choice(available_fingers)
        else:
            self.current_target_finger = self.exemplary_sequence.pop(0)
            self.goal_sequence.append(self.current_target_finger)
            self.exemplary_sequence.append(self.current_target_finger)

        # make one hot encoding
        goal = np.zeros(len(FINGERTIP_SITE_NAMES))
        goal[self.current_target_finger] = 1

        self.goal_sequence.append(self.current_target_finger)

        return goal

    def get_target_finger_position(self) -> np.ndarray:
        return self.sim.data.get_site_xpos(FINGERTIP_SITE_NAMES[self.current_target_finger]).flatten()

    def _is_success(self,
                    achieved_goal,
                    desired_goal):
        d = get_fingertip_distance(self.get_thumb_position(),
                                   self.get_finger_position(FINGERTIP_SITE_NAMES[self.current_target_finger]))
        return (d < self.distance_threshold).astype(np.float32)

    def reset(self):
        self.goal_sequence = []
        self.exemplary_sequence = self.default_exemplary_sequence.copy()

        ret = super().reset()
        return ret, {}

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

    def step(self,
             action):
        observation, reward, done, truncated, info = super().step(action)

        # set next subgoal if current one is achieved
        if info["is_success"]:
            print(
                f"Reached Target {self.current_target_finger} (after reaching {len(self.goal_sequence) - 1} targets)!")
            self.goal = self._sample_goal()

        return observation, reward, done, truncated, info
