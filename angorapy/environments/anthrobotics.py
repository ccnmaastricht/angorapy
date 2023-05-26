import abc
from abc import ABC
from collections import OrderedDict
from os import path
from typing import Optional, Union, Callable, Dict, Any

import numpy as np
import mujoco

import gym
from gym import error, logger, spaces

from angorapy.common.const import VISION_WH, N_SUBSTEPS
from angorapy.common.senses import Sensation
from angorapy.configs.reward_config import resolve_config_name
from angorapy.environments.utils import mj_qpos_dict_to_qpos_vector, robot_get_obs
from angorapy.common import reward


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
    """Base class for all anthropomorphic environments.

    Connects to a MuJoCo model and its simulation and provides basic functionality for an anthropomorphic environment.

    Args:
        model_path (str): Path to the model xml file.
        frame_skip (int): Number of frames to skip per action.
        initial_qpos (dict): Dictionary of initial joint positions.
        vision (bool): Whether to use vision or not.
        touch (bool): Whether to use touch or not.
        render_mode (str): The render mode to use, in ["human", "rgb_array"].
        camera_id (int): The camera id to use.
        camera_name (str): The camera name to use.
        delta_t (float): The time between two control steps.
        n_substeps (int): The number of simulation steps per control step.

    Attributes:
        metadata (dict): The metadata of the environment.
        observation_space (gym.spaces.Space): The observation space of the environment.
        action_space (gym.spaces.Space): The action space of the environment.
        reward_range (tuple): The reward range of the environment.
        spec (gym.envs.registration.EnvSpec): The specification of the environment.
        viewer (mujoco_py.MjViewer): The viewer of the environment, used for rendering in different modes.
        model (mujoco_py.MjModel): The MuJoCo model used for simulation.
        data (mujoco_py.MjData): The MuJoCo data object containing simulation data.
        initial_state (dict): The initial state of the environment.
        goal (np.ndarray): The goal of the environment.
        vision (bool): Whether to use vision or not.
        touch (bool): Whether to use touch or not.
        render_mode (str): The render mode to use, in ["human", "rgb_array"].
        camera_id (int): The camera id to use.
        camera_name (str): The camera name to use.
        frame_skip (int): Number of frames to skip per action.
    """
    metadata: Dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            model_path,
            frame_skip,
            initial_qpos=None,
            vision=False,
            touch=True,
            render_mode: Optional[str] = None,
            camera_id: Optional[int] = None,
            camera_name: Optional[str] = None,
            delta_t: float = 0.002,
            n_substeps: int = N_SUBSTEPS,
    ):
        # mujoco model
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        self.model = mujoco.MjModel.from_xml_path(filename=fullpath)
        self.model.vis.global_.offwidth = VISION_WH
        self.model.vis.global_.offheight = VISION_WH
        self.data = mujoco.MjData(self.model)

        # time control
        self._delta_t_control: float = delta_t
        self._delta_t_simulation: float = delta_t
        self._simulation_steps_per_control_step: int = int(self._delta_t_control // self._delta_t_simulation)

        # rendering
        self.render_mode = render_mode
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.frame_skip = frame_skip

        self._viewers = {}
        self.viewer = None

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # senses
        self.vision = vision
        self.touch = touch

        # store initial state
        if initial_qpos is None or not initial_qpos:
            initial_qpos = self.data.qpos[:]
        else:
            initial_qpos = mj_qpos_dict_to_qpos_vector(self.model, initial_qpos)

        self.initial_state = {
            "qpos": initial_qpos,
            "qvel": self.data.qvel[:]
        }

        # setup environment
        self._env_setup(initial_state=self.initial_state)
        self._set_action_space()

        self.goal = np.array([])
        observation = self._get_obs()
        self._set_observation_space(observation)

        self.model.opt.timestep = delta_t
        self.original_n_substeps = n_substeps

        self._set_default_reward_function_and_config()

    # SPACES
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    # REWARD
    @abc.abstractmethod
    def _set_default_reward_function_and_config(self):
        pass

    def set_reward_function(self, function: Union[str, Callable]):
        """Set the environment reward function by its config identifier or a callable."""
        if isinstance(function, str):
            try:
                function = getattr(reward, function.split(".")[0])
            except AttributeError:
                raise AttributeError("Reward function unknown.")
        elif isinstance(function, Callable):
            pass
        else:
            raise ValueError("Unknown format for given reward function. Provide a string or a Callable.")

        self.reward_function = function

    def set_reward_config(self, new_config: Union[str, dict]):
        """Set the environment's reward configuration by its identifier or a dict."""
        if isinstance(new_config, str):
            new_config: dict = resolve_config_name(new_config)

        self.reward_config = new_config
        if "SUCCESS_DISTANCE" in self.reward_config.keys():
            self.distance_threshold = self.reward_config["SUCCESS_DISTANCE"]

        self.assert_reward_setup()

    @abc.abstractmethod
    def assert_reward_setup(self):
        pass

    def compute_reward(self, achieved_goal, goal, info):
        """Compute reward with additional success bonus."""
        return self.reward_function(self, achieved_goal, goal, info)

    @abc.abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        pass

    @abc.abstractmethod
    def _get_achieved_goal(self):
        pass

    # TIME
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def set_delta_t_simulation(self, new: float):
        """Set new value for the simulation delta t."""
        assert np.isclose(self._delta_t_control % new, 0, rtol=1.e-3, atol=1.e-4), \
            f"Delta t of simulation must divide control delta t into integer " \
            f"parts, but gives {self._delta_t_control % new}."

        self._delta_t_simulation = new
        self.model.opt.timestep = self._delta_t_simulation

        self._simulation_steps_per_control_step = int(self._delta_t_control / self._delta_t_simulation * self.original_n_substeps)

    def set_original_n_substeps_to_sspcs(self):
        self.original_n_substeps = self._simulation_steps_per_control_step

    # SETUP
    def _env_setup(self, initial_state):
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.reset_model()

        obs = self._get_obs()["observation"]

        if self.render_mode == "human":
            self.render()

        return obs, {}

    # CONTROL
    def _set_action(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self.data.ctrl[:] = action

    def do_simulation(self, ctrl, n_frames):
        self._set_action(ctrl)

        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
            self._step_callback()

        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    # RENDERING
    def _get_viewer(
        self, mode
    ) -> Union[
        "gym.envs.mujoco.mujoco_rendering.Viewer",
        "gym.envs.mujoco.mujoco_rendering.RenderContextOffscreen",
    ]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                from gym.envs.mujoco.mujoco_rendering import Viewer

                self.viewer = Viewer(self.model, self.data)
            elif mode in {"rgb_array", "depth_array"}:
                from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode in {
            "rgb_array",
            "depth_array",
        }:
            camera_id = self.camera_id
            camera_name = self.camera_name

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

                self._get_viewer(self.render_mode).render(camera_id=camera_id)

        if self.render_mode == "rgb_array":
            data = self._get_viewer(self.render_mode).read_pixels(depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif self.render_mode == "depth_array":
            self._get_viewer(self.render_mode).render()
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(self.render_mode).read_pixels(depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif self.render_mode == "human":
            self._get_viewer(self.render_mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    # SIMULATION STATE
    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

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

    def _reset_sim(self):
        """Reset the simulation."""
        self.reset_model()
        return True

    def reset_model(self):
        self.set_state(**self.initial_state)

    # SENSES
    def get_proprioception(self):
        """Get proprioception sensor readings."""
        robot_qpos, robot_qvel = robot_get_obs(self.model, self.data)
        proprioception = np.concatenate([robot_qpos, robot_qvel])

        return proprioception

    def get_touch(self):
        """Get touch sensor readings."""
        touch = self.data.sensordata[self._touch_sensor_id]

        return touch

    def get_vision(self):
        """Get vision sensor readings."""
        vision = self.render("rgb_array", VISION_WH, VISION_WH) if self.vision else np.array([])

        return vision

    def _get_obs(self):
        """Get proprioception, touch and vision sensor readings."""
        return {
            'observation': Sensation(
                proprioception=self.get_proprioception(),
                touch=self.get_touch() if self.touch else None,
                vision=self.get_vision(),
                goal=self.goal.copy()
            ),

            'desired_goal': self.goal.copy(),
            'achieved_goal': self._get_achieved_goal().ravel(),
        }