#!/usr/bin/env python
"""ShadowHand Baseclass for NRP Environments."""

import abc
import copy
import os
import random

import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.utils import seeding

FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]

DEFAULT_INITIAL_QPOS = {
    'robot0:WRJ1': -0.16514339750464327,
    'robot0:WRJ0': -0.31973286565062153,
    'robot0:FFJ3': 0.14340512546557435,
    'robot0:FFJ2': 0.32028208333591573,
    'robot0:FFJ1': 0.7126053607727917,
    'robot0:FFJ0': 0.6705281001412586,
    'robot0:MFJ3': 0.000246444303701037,
    'robot0:MFJ2': 0.3152655251085491,
    'robot0:MFJ1': 0.7659800313729842,
    'robot0:MFJ0': 0.7323156897425923,
    'robot0:RFJ3': 0.00038520700007378114,
    'robot0:RFJ2': 0.36743546201985233,
    'robot0:RFJ1': 0.7119514095008576,
    'robot0:RFJ0': 0.6699446327514138,
    'robot0:LFJ4': 0.0525442258033891,
    'robot0:LFJ3': -0.13615534724474673,
    'robot0:LFJ2': 0.39872030433433003,
    'robot0:LFJ1': 0.7415570009679252,
    'robot0:LFJ0': 0.704096378652974,
    'robot0:THJ4': 0.003673823825070126,
    'robot0:THJ3': 0.5506291436028695,
    'robot0:THJ2': -0.014515151997119306,
    'robot0:THJ1': -0.0015229223564485414,
    'robot0:THJ0': -0.7894883021600622,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../assets/hand/', 'shadowhand.xml')


def generate_random_sim_qpos(base: dict) -> dict:
    """Generate a random state of the simulation."""
    for key, val in base.items():
        if key in ["robot0:WRJ1", "robot0:WRJ0"]:
            continue  # do not randomize the wrist

        base[key] = val + val * random.gauss(0, 0.1)

    return base


def get_palm_position(sim):
    """Return the robotic hand'serialization palm'serialization center."""
    palm_idx = sim.model.body_names.index("robot0:palm")
    return np.array(sim.model.body_pos[palm_idx])


def get_fingertip_distance(ft_a, ft_b):
    """Return the distance between two vectors representing finger tip positions."""
    assert ft_a.shape == ft_b.shape
    return np.linalg.norm(ft_a - ft_b, axis=-1)


class BaseNRPShadowHandEnv(gym.GoalEnv, abc.ABC):
    """Base class for all shadow hand environments, setting up mostly visual characteristics of the environment."""

    def __init__(self, initial_qpos, distance_threshold, n_substeps=20, relative_control=True):
        gym.utils.EzPickle.__init__(**locals())

        self.relative_control = relative_control
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.distance_threshold = distance_threshold
        self.reward_type = "dense"

        model = mujoco_py.load_model_from_path(MODEL_PATH)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self._viewers = {}
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        for k, v in self.sim.model._sensor_name2id.items():
            if 'robot0:TS_' in k:
                self._touch_sensor_id_site_id.append(
                    (v, self.sim.model._site_name2id[k.replace('robot0:TS_', 'robot0:T_')]))
                self._touch_sensor_id.append(v)

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()

        self.action_space = spaces.Box(-1., 1., shape=(20,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Dict(
                {name: spaces.Box(-np.inf, np.inf, shape=val.shape, dtype="float32")
                 for name, val in obs['observation'].dict().items()}
            ),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # INFROMATION METHODS
    def get_fingertip_positions(self):
        """Get positions of all fingertips in euclidean space. Each position is encoded by three floating point numbers,
        as such the output is a 15-D numpy array."""
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    # ENV METHODS

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(BaseNRPShadowHandEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=500, height=500):
        self._render_callback(render_targets=(mode == "human"))
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)

            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    @abc.abstractmethod
    def _get_obs(self):
        pass

    def _set_action(self, action):
        assert action.shape == (20,)

        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        if self.relative_control:
            actuation_center = np.zeros_like(action)
            for i in range(self.sim.data.ctrl.shape[0]):
                actuation_center[i] = self.sim.data.get_joint_qpos(
                    self.sim.model.actuator_names[i].replace(':A_', ':'))
            for joint_name in ['FF', 'MF', 'RF', 'LF']:
                act_idx = self.sim.model.actuator_name2id(
                    'robot0:A_{}J1'.format(joint_name))
                actuation_center[act_idx] += self.sim.data.get_joint_qpos(
                    'robot0:{}J0'.format(joint_name))
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    @abc.abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        pass

    @abc.abstractmethod
    def _sample_goal(self):
        pass

    @abc.abstractmethod
    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation."""
        pass

    @abc.abstractmethod
    def _render_callback(self, render_targets=False):
        """A custom callback that is called before rendering. Can be used to implement custom visualizations."""
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:palm')
        lookat = self.sim.data.body_xpos[body_id]

        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        # set colors
        self.sim.model.mat_rgba[2] = np.array([16, 18, 35, 255]) / 255  # hand
        # self.sim.model.mat_rgba[2] = np.array([200, 200, 200, 255]) / 255  # hand
        self.sim.model.mat_rgba[4] = np.array([71, 116, 144, 255]) / 255  # background
        # self.sim.model.geom_rgba[48] = np.array([0.5, 0.5, 0.5, 0])

        self.viewpoint = "topdown"

        if self.viewpoint == "topdown":
            # rotate camera to top down view
            self.viewer.cam.distance = 0.32  # zoom in
            self.viewer.cam.azimuth = -90.0  # wrist to the bottom
            self.viewer.cam.elevation = -90.0  # top down view
            self.viewer.cam.lookat[1] -= 0.07  # slightly move forward
        elif self.viewpoint == "side":
            # rotate camera to side view
            self.viewer.cam.distance = 0.35  # zoom in
            self.viewer.cam.azimuth = 25.0  # top down view
            self.viewer.cam.elevation = -45.0  # top down view
            self.viewer.cam.lookat[1] -= 0.04  # slightly move forward
        else:
            raise NotImplementedError("Unknown Viewpoint.")


if __name__ == "__main__":
    from environments import *

    # env = gym.make("HandTappingAbsolute-v1")
    # env = gym.make("HandFreeReachLFAbsolute-v0")
    # env = gym.make("BaseShadowHandEnv-v0")
    # env = gym.make("HandManipulateBlock-v0")
    env = gym.make("HandReachDenseRelative-v0")
    d, s = False, env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        s, r, d, i = env.step(action)
        if d:
            env.reset()
