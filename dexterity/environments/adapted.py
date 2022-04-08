from gym import spaces
from gym.envs.box2d import LunarLanderContinuous
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
import numpy as np


class InvertedPendulumNoVelEnv(InvertedPendulumEnv):
    """InvertedPendulum Environment without velocities directly encoded in the state."""

    def _get_obs(self):
        return self.sim.data.qpos.ravel()


class ReacherNoVelEnv(ReacherEnv):
    """InvertedPendulum Environment without velocities directly encoded in the state."""

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            # self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])


class HalfCheetahNoVelEnv(HalfCheetahEnv):
    """HalfCheetah Environment without velocities directly encoded in the state."""

    def _get_obs(self):
        return self.sim.data.qpos.flat[1:]


class LunarLanderContinuousNoVel(LunarLanderContinuous):
    """LunarLander Continuous Environment without velocities directly encoded in the state."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.reset()

    def step(self, action):
        o, r, d, info = super().step(action)
        o = np.concatenate([o[0:2], [o[4]], o[6:]])

        return o, r, d, info


class LunarLanderMultiDiscrete(LunarLanderContinuous):
    """LunarLander Continuous Environment but again made discrete."""

    discrete_bin_count = 10

    def __init__(self):
        self.discrete_action_values = np.linspace(-1, 1, LunarLanderMultiDiscrete.discrete_bin_count)
        super().__init__()
        self.action_space = spaces.MultiDiscrete(np.ones(2) * LunarLanderMultiDiscrete.discrete_bin_count)

        self.reset()

    def step(self, action: np.ndarray):
        action = self.discrete_action_values[action.astype(np.int)]

        return super().step(action)