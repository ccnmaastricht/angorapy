#!/usr/bin/env python
"""BaseShadowHandEnv Environment Wrappers."""
import abc
import copy
import os
import random
from typing import Callable, Union, Optional

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from angorapy.common import reward

from angorapy.common.const import N_SUBSTEPS
from angorapy.configs.reward_config import resolve_config_name
from angorapy.environments.anthrobotics import AnthropomorphicEnv
from angorapy.environments.utils import mj_get_category_names
from angorapy.utilities.util import mpi_print

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
# MODEL_PATH_MANIPULATE = os.path.join(os.path.dirname(__file__), 'assets/hand/', 'shadowhand.xml')
MODEL_PATH_MANIPULATE = os.path.join(os.path.dirname(__file__), '../assets/hand/', 'shadowhand_manipulate.xml')


def generate_random_sim_qpos(base: dict) -> dict:
    """Generate a random state of the simulation."""
    for key, val in base.items():
        if key in ["robot0:WRJ1", "robot0:WRJ0"]:
            continue  # do not randomize the wrist

        base[key] = val + val * random.gauss(0, 0.1)

    return base


def get_palm_position(model):
    """Return the robotic hand's palm's center."""
    return model.body("robot0:palm").pos


def get_fingertip_distance(ft_a, ft_b):
    """Return the distance between two vectors representing finger tip positions."""
    assert ft_a.shape == ft_b.shape
    return np.linalg.norm(ft_a - ft_b, axis=-1)


class BaseShadowHandEnv(AnthropomorphicEnv):  #, abc.ABC):
    """Base class for all shadow hand environments, setting up mostly visual characteristics of the environment."""

    continuous = True
    discrete_bin_count = 11

    def __init__(self,
                 initial_qpos,
                 distance_threshold,
                 n_substeps=N_SUBSTEPS,
                 delta_t=0.002,
                 relative_control=True,
                 model=MODEL_PATH,
                 vision=False,
                 render_mode: Optional[str] = None
                 ):
        gym.utils.EzPickle.__init__(**locals())

        self.relative_control = relative_control
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.distance_threshold = distance_threshold
        self.reward_type = "dense"
        self._freeze_wrist = False
        self.color_scheme = "default"
        self.viewpoint = "topdown"

        super(BaseShadowHandEnv, self).__init__(model_path=model, frame_skip=n_substeps, initial_qpos=initial_qpos,
                                                vision=vision, render_mode=render_mode, delta_t=delta_t, n_substeps=n_substeps)

        self.seed()
        self.initial_state = copy.deepcopy(self.get_state())

        obs = self._get_obs()

        self.viewer_setup()

    def _set_action_space(self):
        if self.continuous:
            self.action_space = spaces.Box(-1., 1., shape=(20,), dtype=float)
        else:
            self.action_space = spaces.MultiDiscrete(np.ones(20) * BaseShadowHandEnv.discrete_bin_count)
            self.discrete_action_values = np.linspace(-1, 1, BaseShadowHandEnv.discrete_bin_count)

    def _set_observation_space(self, obs):
        # bounds are set to max of dtype to avoid infinity warnings
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=obs['achieved_goal'].shape),
            achieved_goal=spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=obs['achieved_goal'].shape),
            observation=spaces.Dict(
                {name: spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=val.shape)
                 for name, val in obs['observation'].dict().items()}
            ),
        ))

    def toggle_wrist_freezing(self):
        """Toggle flag preventing the wrist from moving."""
        self._freeze_wrist = not self._freeze_wrist
        print("Wrist movements are now frozen.")

    # INFORMATION METHODS
    def get_fingertip_positions(self):
        """Get positions of all fingertips in euclidean space. Each position is encoded by three floating point numbers,
        as such the output is a 15-D numpy array."""
        goal = [self.data.site(name).xpos.flatten() for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    # ENV METHODS

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray):
        if self.continuous:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = self.discrete_action_values[action.astype(int)]

        if self._freeze_wrist:
            action[:2] = 0

        # perform simulation
        self.do_simulation(action, n_frames=self.original_n_substeps)

        # read out observation from simulation
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": obs["desired_goal"]
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, info

    def reset(self, **kwargs):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(BaseShadowHandEnv, self).reset(**kwargs)
        did_reset_sim = False

        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()

        obs = self._get_obs()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": obs["desired_goal"]
        }

        return obs, info

    def reset_model(self):
        self.set_state(**self.initial_state)

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self):
        return super().render()

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.reset_model()

        return True

    @abc.abstractmethod
    def _get_obs(self):
        pass

    def _set_action(self, action):
        assert action.shape == (20,)

        actuator_names = mj_get_category_names(self.model, "actuator")
        ctrlrange = self.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        if self.relative_control:
            actuation_center = np.zeros_like(action)
            for i in range(self.data.ctrl.shape[0]):
                actuation_center[i] = self.data.jnt(actuator_names[i].replace(b':A_', b':')).qpos
            for joint_name in ['FF', 'MF', 'RF', 'LF']:
                act_idx = actuator_names.index(str.encode(f'robot0:A_{joint_name}J1'))
                actuation_center[act_idx] += self.data.jnt(str.encode(f'robot0:{joint_name}J0')).qpos
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

        self.data.ctrl[:] = actuation_center + action * actuation_range
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    @abc.abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        pass

    @abc.abstractmethod
    def _sample_goal(self):
        pass

    def _env_setup(self, initial_state):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation."""
        for k, v in zip(mj_get_category_names(self.model, "sensor"), self.model.sensor_adr):
            if b'robot0:TS_' in k:
                # self._touch_sensor_id_site_id.append((v, self.model._site_name2id[k.replace('robot0:TS_', 'robot0:T_')]))
                self._touch_sensor_id.append(v)

        if initial_state is not None:
            self.set_state(**initial_state)

    @abc.abstractmethod
    def _render_callback(self, render_targets=False):
        """A custom callback that is called before rendering. Can be used to implement custom visualizations."""
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def change_perspective(self, perspective: str):
        assert perspective in ["topdown", "side", "topdown-far"], "This viewpoint has no settings available."

        self.viewpoint = perspective
        self.viewer_setup()

    def change_color_scheme(self, color_scheme: str):
        assert color_scheme in ["default", "inverted"]

        self.color_scheme = color_scheme
        self.viewer_setup()

    def viewer_setup(self):
        # lookat = get_palm_position(self.model)
        #
        # for idx, value in enumerate(lookat):
        #     self.viewer.cam.lookat[idx] = value

        # hand color

        if self.color_scheme == "default":
            self.model.mat_rgba[2] = np.array([29, 33, 36, 255]) / 255  # hand
            self.model.mat_rgba[4] = np.array([255, 255, 255, 255]) / 255  # background
        elif self.color_scheme == "inverted":
            self.model.mat_rgba[2] = np.array([200, 200, 200, 255]) / 255  # hand
            self.model.mat_rgba[4] = np.array([0, 0, 0, 255]) / 255  # background
        else:
            raise NotImplementedError(f"Unknown Color Scheme {self.color_scheme}.")

        # self.sim.model.mat_rgba[4] = np.array([159, 41, 54, 255]) / 255  # background
        # self.sim.model.geom_rgba[48] = np.array([0.5, 0.5, 0.5, 0])

        if self.viewer is None:
            return

        mpi_print("Setting up camera.")

        if self.viewpoint == "topdown":
            # rotate camera to top down view
            self.viewer.cam.distance = 0.30  # zoom in
            self.viewer.cam.azimuth = -90.0  # wrist to the bottom
            self.viewer.cam.elevation = -90.0  # top down view
            self.viewer.cam.lookat[1] += 0.03  # slightly move forward
        elif self.viewpoint == "topdown-far":
            # rotate camera to top down view
            self.viewer.cam.distance = 0.4  # zoom in
            self.viewer.cam.azimuth = -90.0  # wrist to the bottom
            self.viewer.cam.elevation = -90.0  # top down view
            self.viewer.cam.lookat[1] += 0.03  # slightly move forward
        elif self.viewpoint == "side":
            # rotate camera to side view
            self.viewer.cam.distance = 0.35  # zoom in
            self.viewer.cam.azimuth = 25.0  # top down view
            self.viewer.cam.elevation = -45.0  # top down view
            self.viewer.cam.lookat[1] -= 0.04  # slightly move forward
        else:
            raise NotImplementedError(f"Unknown Viewpoint {self.viewpoint}.")

        mpi_print("Camera setup finished.")


if __name__ == '__main__':
    hand = BaseShadowHandEnv(
        initial_qpos=DEFAULT_INITIAL_QPOS,
        distance_threshold=0.2
    )
    pass
    print()
