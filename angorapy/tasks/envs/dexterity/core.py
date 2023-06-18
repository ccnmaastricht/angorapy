#!/usr/bin/env python
"""BaseShadowHandEnv Environment Wrappers."""
import abc
import copy
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from angorapy.common.const import N_SUBSTEPS
from angorapy.tasks.core import AnthropomorphicEnv
from angorapy.tasks.envs.dexterity.consts import FINGERTIP_SITE_NAMES
from angorapy.tasks.envs.dexterity.mujoco_model.robot import ShadowHand
from angorapy.tasks.utils import mj_get_category_names
from angorapy.utilities.util import mpi_print


class BaseShadowHandEnv(AnthropomorphicEnv, abc.ABC):
    """Base class for all shadow hand environments, setting up mostly visual characteristics of the environment."""

    continuous = True
    discrete_bin_count = 11

    def __init__(self,
                 initial_qpos=None,
                 n_substeps=N_SUBSTEPS,
                 delta_t=0.002,
                 relative_control=True,
                 model=None,
                 vision=False,
                 touch=True,
                 render_mode: Optional[str] = None):
        gym.utils.EzPickle.__init__(**locals())

        if model is None:
            model = ShadowHand()

        self.relative_control = relative_control
        self._thumb_touch_sensor_id = []
        self.reward_type = "dense"
        self._freeze_wrist = False
        self.color_scheme = "default"
        self.viewpoint = "topdown"
        self.thumb_name = 'robot/S_thtip'
        self.palm_name = 'robot/rh_palm'

        super(BaseShadowHandEnv,
              self).__init__(model=model,
                             frame_skip=n_substeps,
                             initial_qpos=initial_qpos,
                             vision=vision,
                             touch=touch,
                             render_mode=render_mode,
                             delta_t=delta_t,
                             n_substeps=n_substeps)

        self.seed()
        self.initial_state = copy.deepcopy(self.get_state())

        obs = self._get_obs()
        self.viewer_setup()

    # SPACES
    def _set_observation_space(self,
                               obs):
        # bounds are set to max of dtype to avoid infinity warnings
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Dict(
                {name: spaces.Box(np.finfo(np.float32).min,
                                  np.finfo(np.float32).max,
                                  shape=val.shape)
                 for name, val in obs['observation'].dict().items()}
            ),
        ))

    # TOGGLES
    def toggle_wrist_freezing(self):
        """Toggle flag preventing the wrist from moving."""
        self._freeze_wrist = not self._freeze_wrist
        print("Wrist movements are now frozen.")

    # GETTERS
    def get_finger_position(self,
                            finger_name: str) -> np.ndarray:
        """Get position of the specified finger in space."""
        return self.data.site(finger_name).xpos.flatten()

    def get_fingertip_positions(self):
        """Get positions of all fingertips in euclidean space. Each position is encoded by three floating point numbers,
        as such the output is a 15-D numpy array."""
        goal = [self.data.site(name).xpos.flatten() for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    def get_thumb_position(self) -> np.ndarray:
        """Get the position of the thumb in space."""
        return self.data.site(self.thumb_name).xpos.flatten()

    def get_palm_position(self) -> np.ndarray:
        """Return the robotic hand's palm's center."""
        return self.model.site("robot/palm_center_site").pos

    def is_thumb_tip_touching(self):
        if sum(self.data.sensordata[self._touch_sensor_id]) > 0.0:
            return True

    def seed(self,
             seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # CONTROL
    def _set_action(self, action):
        assert action.shape == (20,), f"Action shape must be (20,) not {action.shape}"

        actuator_names, actuator_adresses = mj_get_category_names(self.model, "actuator")
        ctrlrange = self.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        if self.relative_control:
            actuation_center = np.zeros_like(action)
            for i in range(self.data.ctrl.shape[0]):
                jnt_or_ten_name = actuator_names[i].replace(b'_A_', b'_')
                if self.model.names.index(jnt_or_ten_name) in self.model.name_jntadr:
                    actuation_center[i] = self.data.jnt(jnt_or_ten_name).qpos
                elif self.model.names.index(jnt_or_ten_name) in self.model.name_tendonadr:
                    actuation_center[i] = self.data.actuator(actuator_names[i]).length
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

        self.data.ctrl[:] = actuation_center + action * actuation_range
        self.data.ctrl[:] = np.clip(self.data.ctrl,
                                    ctrlrange[:, 0],
                                    ctrlrange[:, 1])

    def step(self, action: np.ndarray):
        if self._freeze_wrist:
            action[:2] = 0

        return super(BaseShadowHandEnv, self).step(action)

    # SIMULATION
    def _env_setup(self, initial_state):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation."""
        for k, v in zip(mj_get_category_names(self.model, "sensor")[0], self.model.sensor_adr):
            if b'robot/TS_' in k:
                self._touch_sensor_id.append(v)

                if b'thtip' in k:
                    self._thumb_touch_sensor_id.append(v)

        if initial_state is not None:
            self.set_state(**initial_state)

    # RENDERING
    def change_perspective(self,
                           perspective: str):
        assert perspective in ["topdown", "side", "topdown-far"], "This viewpoint has no settings available."

        self.viewpoint = perspective
        self.viewer_setup()

    def change_color_scheme(self,
                            color_scheme: str):
        assert color_scheme in ["default", "inverted"]

        self.color_scheme = color_scheme
        self.viewer_setup()

    def _get_info(self):
        return super(BaseShadowHandEnv, self)._get_info()

    def viewer_setup(self):
        mpi_print(f"Setting up color scheme {self.color_scheme}.")

        # hand color
        if self.color_scheme == "default":
            pass
            # self.model.mat_rgba[2] = np.array([30, 30, 30, 255]) / 255  # hand
        elif self.color_scheme == "inverted":
            self.model.mat_rgba[0] = np.array([200, 200, 200, 255]) / 255  # hand
            self.model.mat_rgba[1] = np.array([0, 0, 0, 255]) / 255  # background
        else:
            raise NotImplementedError(f"Unknown Color Scheme {self.color_scheme}.")

        # self.sim.model.mat_rgba[4] = np.array([159, 41, 54, 255]) / 255  # background
        # self.sim.model.geom_rgba[48] = np.array([0.5, 0.5, 0.5, 0])

        if self.viewer is None:
            return

        mpi_print(f"Setting up camera with viewpoint '{self.viewpoint}'.")

        if self.viewpoint == "topdown":
            # rotate camera to top down view
            self.viewer.cam.distance = 0.5  # zoom in
            self.viewer.cam.azimuth = 0.0  # wrist to the bottom
            self.viewer.cam.elevation = -90.0  # top down view
            self.viewer.cam.lookat[0] += 0.3  # slightly move forward
        elif self.viewpoint == "topdown-far":
            # rotate camera to top down view
            self.viewer.cam.distance = 0.7  # increased values zoom out
            self.viewer.cam.azimuth = -0.0  # wrist to the bottom
            self.viewer.cam.elevation = -90.0  # top down view
            self.viewer.cam.lookat[0] += 0.25  # slightly move forward
        elif self.viewpoint == "side":
            # rotate camera to side view
            self.viewer.cam.distance = 0.35  # zoom in
            self.viewer.cam.azimuth = 25.0  # top down view
            self.viewer.cam.elevation = -45.0  # top down view
            self.viewer.cam.lookat[0] -= 0.04  # slightly move forward
        elif self.viewpoint == "angled":
            self.viewer.cam.distance = 0.5  # zoom in
            self.viewer.cam.azimuth = 35.0  # wrist to the bottom
            self.viewer.cam.elevation = -125.0  # top down view
            self.viewer.cam.lookat[0] += 0.  # slightly move forward
            self.viewer.cam.lookat[1] += 0.0  # slightly move forward
        else:
            raise NotImplementedError(f"Unknown Viewpoint {self.viewpoint}.")

        mpi_print("Camera setup finished.")


if __name__ == '__main__':
    class TestShadowHandEnv(BaseShadowHandEnv):
        def _set_default_reward_function_and_config(self):
            pass

        def assert_reward_setup(self):
            pass

        def _sample_goal(self):
            return np.zeros(1)

    test_env = TestShadowHandEnv()
    test_env.reset()