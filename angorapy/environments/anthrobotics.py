from abc import ABC
from collections import OrderedDict
from os import path
from typing import Optional

import numpy as np
import mujoco

import gym
from gym import error, logger, spaces

from angorapy.common.const import VISION_WH
from angorapy.environments.utils import mj_qpos_dict_to_qpos_vector


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class AnthropomorphicEnv(gym.Env, ABC):
    """Superclass for all environments."""

    def __init__(self, model_path, frame_skip, initial_qpos=None, vision=False):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        self.model = mujoco.MjModel.from_xml_path(filename=fullpath)
        self.data = mujoco.MjData(self.model)
        self._viewers = {}
        self.viewer = None
        self.vision = vision

        self.frame_skip = frame_skip
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        if initial_qpos is None or not initial_qpos:
            initial_qpos = self.data.qpos[:]
        else:
            initial_qpos = mj_qpos_dict_to_qpos_vector(self.model, initial_qpos)

        self.initial_state = {
            "qpos": initial_qpos,
            "qvel": self.data.qvel[:]
        }

        self._env_setup(initial_state=self.initial_state)
        self._set_action_space()

        action = self.action_space.sample()
        observation = self._get_obs()
        self._set_observation_space(observation)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _env_setup(self, initial_state):
        raise NotImplementedError

    # methods to override:
    # ----------------------------

    def _get_obs(self):
        raise NotImplementedError

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

    # -----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        if not return_info:
            return ob
        else:
            return ob, {}

    def get_state(self):
        """Get the current state of the simulation."""
        return {
            "qpos": self.data.qpos[:],
            "qvel": self.data.qvel[:],
        }

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,), \
            f"State shape [{qpos.shape}|{qvel.shape}] does not fit [{self.model.nq}|{self.model.nv}]"

        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self._set_action(ctrl)

        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
            self._step_callback()

        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _set_action(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self.data.ctrl[:] = action

    def _step_callback(self):
        pass

    def render(
        self,
        mode="human",
        width=VISION_WH,
        height=VISION_WH,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode, width=VISION_WH, height=VISION_WH):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                from gym.envs.mujoco.mujoco_rendering import Viewer

                self.viewer = Viewer(self.model, self.data)
            elif mode == "rgb_array" or mode == "depth_array":
                from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(
                    width, height, self.model, self.data
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
