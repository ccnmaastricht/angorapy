"""NRP implementations of reaching tasks."""
import numpy as np
from gym.envs.robotics.hand.reach import DEFAULT_INITIAL_QPOS, HandReachEnv
from gym.envs.robotics.utils import robot_get_obs


class NRPShadowHandReach(HandReachEnv):
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
