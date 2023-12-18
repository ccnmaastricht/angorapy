import abc
from abc import ABC
from os import path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import dm_control
import gymnasium as gym
import mujoco
import numpy as np
from dm_control.mjcf import RootElement
from gymnasium import spaces
from gymnasium.utils import seeding
from mujoco import MjModel

from angorapy.common.const import N_SUBSTEPS
from angorapy.common.const import VISION_WH
from angorapy.common.senses import Sensation
from angorapy.tasks.reward_config import resolve_config_name
from angorapy.tasks.utils import convert_observation_to_space
from angorapy.tasks.utils import mj_get_category_names
from angorapy.tasks.utils import mj_qpos_dict_to_qpos_vector
from angorapy.tasks.world_building.entities import _Entity


def _parse_model_definition(model_definition: Union[str, RootElement, _Entity]) -> Tuple[MjModel, RootElement]:
    """Parse a variable model definition into a MuJoCo model and a MJCF Model.

    Possible model definitions are:
        - a path to a model xml file
        - a dm_control.mjcf.RootElement object
        - a angorapy.tasks.world_building.entities._Entity object

    Args:
        model_definition (Union[str, RootElement, _Entity]): The model definition to parse.

    Returns:
        Tuple[MjModel, RootElement]: The parsed MuJoCo model and MuJoCo XML model.
    """
    if isinstance(model_definition, str):
        model_path = model_definition
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(path.dirname(__file__), "models", model_path)

        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        model = mujoco.MjModel.from_xml_path(filename=fullpath)
        mjcf_model = dm_control.mjcf.from_path(fullpath)
    elif isinstance(model_definition, dm_control.mjcf.RootElement):
        model = mujoco.MjModel.from_xml_string(xml=model_definition.to_xml_string(),
                                               assets=model_definition.get_assets())
        mjcf_model = model_definition
    elif isinstance(model_definition, _Entity):
        model = mujoco.MjModel.from_xml_string(xml=model_definition.mjcf_model.to_xml_string(),
                                               assets=model_definition.mjcf_model.get_assets())
        mjcf_model = model_definition._mjcf_root.root_model
    else:
        raise ValueError(f"model must be either a path to a model xml file or mjcf.RootElement, "
                         f"not {type(model_definition)}")

    return model, mjcf_model


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
    metadata: Dict[str, Any] = {
        "render_modes": ["human", "rgb_array"]}

    continuous = True
    discrete_bin_count = 11

    def __init__(
            self,
            model: Union[str, RootElement, _Entity],
            initial_qpos=None,
            vision=False,
            touch=True,
            relative_control=True,
            render_mode: Optional[str] = None,
            camera_id: Optional[int] = 0,
            camera_name: Optional[str] = None,
            delta_t: float = 0.002,
            frame_skip=N_SUBSTEPS,
            n_substeps: int = N_SUBSTEPS,
    ):
        self.model, self.mjcf_model = _parse_model_definition(model)

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

        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []

        # store initial state
        initial_qpos = self.set_initial_qpos(initial_qpos)
        self.initial_state = {
            "qpos": initial_qpos,
            "qvel": self.data.qvel[:]
        }

        # setup environment
        self.relative_control = relative_control
        self._env_setup(initial_state=self.initial_state)
        self._set_action_space()

        self.goal = self._sample_goal()
        observation = self._get_obs()
        self._set_observation_space(observation)

        self.model.opt.timestep = delta_t
        self.original_n_substeps = n_substeps

        self._set_default_reward_function_and_config()
        self.seed()
        self.viewer_setup()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # SPACES
    def _set_action_space(self):
        """Sets the action space of the environment.

        Action spaces may be continuous or discrete. If continuous, the action space is a Box space with the bounds
        given by the model's actuator control range. If discrete, the action space is a MultiDiscrete space splitting
        the continuous action space into a number of bins (default 11)."""

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T

        if self.continuous:
            self.action_space = spaces.Box(-1., 1., shape=(self.model.actuator_ctrlrange.shape[0],), dtype=float)
        else:
            self.action_space = spaces.MultiDiscrete(
                np.ones(self.model.actuator_ctrlrange.shape[0]) * self.discrete_bin_count
            )
            self.discrete_action_values = np.linspace(-1, 1, self.discrete_bin_count)

        return self.action_space

    def _set_observation_space(self,
                               observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    # REWARD
    @abc.abstractmethod
    def _set_default_reward_function_and_config(self):
        """Sets the default reward function and config.

        The reward config is optional (can be empty) but can be used to
        configure the reward function. The reward function is a function that takes the environment and the steps info
        dictionary and returns a reward value."""
        pass

    def set_reward_function(self,
                            function: Callable):
        """Set the environment reward function."""
        self.reward_function = function

    def set_reward_config(self,
                          new_config: Union[str, dict]):
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

    def compute_reward(self, info):
        """Compute reward with additional success bonus."""
        return self.reward_function(self, info)

    def _is_success(self, achieved_goal, desired_goal):
        return False

    # TIME
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def set_delta_t_simulation(self,
                               new: float):
        """Set new value for the simulation delta t."""
        assert np.isclose(self._delta_t_control % new, 0, rtol=1.e-3, atol=1.e-4), \
            f"Delta t of simulation must divide control delta t into integer " \
            f"parts, but gives {self._delta_t_control % new}."

        self._delta_t_simulation = new
        self.model.opt.timestep = self._delta_t_simulation

        self._simulation_steps_per_control_step = int(self._delta_t_control / self._delta_t_simulation *
                                                      self.original_n_substeps)

    def set_original_n_substeps_to_sspcs(self):
        self.original_n_substeps = self._simulation_steps_per_control_step

    # SETUP
    @abc.abstractmethod
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

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.goal = self._sample_goal()
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    # CONTROL
    def step(self, action: np.ndarray):
        if self.continuous:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = self.discrete_action_values[action.astype(int)]

        # perform simulation
        self.do_simulation(action, n_frames=self.original_n_substeps)

        # read out observation from simulation
        obs = self._get_obs()

        done = False
        info = self._get_info()

        reward = self.compute_reward(info)

        if self.render_mode == "human":
            self._render_callback()
            self.render()

        return obs, reward, done, False, info

    def _set_action(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError(f"Action dimension mismatch [{np.array(action).shape}|{self.action_space.shape}]!")

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

    def do_simulation(self, ctrl, n_frames):
        self._set_action(ctrl)

        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
            self._mj_step_callback()

        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _mj_step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    # RENDERING
    def _get_viewer(
            self,
            mode
    ) -> Union[
        "gym.envs.mujoco.mujoco_rendering.Viewer",
        "gym.envs.mujoco.mujoco_rendering.RenderContextOffscreen",
    ]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer

                self.viewer = WindowViewer(self.model, self.data)
            elif mode in {"rgb_array", "depth_array"}:
                from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

                self.viewer = OffScreenViewer(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def render(self):
        if self.render_mode is None:
            return

        self._render_callback()

        if self.render_mode in {
            "rgb_array",
            "depth_array",
        }:
            # camera_id = self.camera_id
            # camera_name = self.camera_name
            #
            # if camera_id is not None and camera_name is not None:
            #     raise ValueError(
            #         "Both `camera_id` and `camera_name` cannot be"
            #         " specified at the same time."
            #     )
            #
            # no_camera_specified = camera_name is None and camera_id is None
            # if no_camera_specified:
            #     camera_name = "track"
            #
            # if camera_id is None:
            #     camera_id = mujoco.mj_name2id(
            #         self.model,
            #         mujoco.mjtObj.mjOBJ_CAMERA,
            #         camera_name,
            #     )

            self._get_viewer(self.render_mode).render(render_mode=self.render_mode)

        if self.render_mode == "rgb_array":
            data = self._get_viewer(self.render_mode).render(camera_id=-1, render_mode=self.render_mode)
            # original image is upside-down, so flip it
            # return data[::-1, :, :]
            return data
        elif self.render_mode == "depth_array":
            self._get_viewer(self.render_mode).render()
            # Extract depth part of the render() tuple
            data = self._get_viewer(self.render_mode).render(depth=True)[1]
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

    def get_robot_state(self):
        """Returns all joint positions and velocities associated with a robot."""
        joint_names, _ = mj_get_category_names(self.model, "jnt")
        if self.data.qpos is not None and joint_names:
            names = [n for n in joint_names if n.startswith(b"robot")]
            return (
                np.array([self.data.jnt(name).qpos for name in names]).flatten(),
                np.array([self.data.jnt(name).qvel for name in names]).flatten(),
            )
        return np.zeros(0), np.zeros(0)

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
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(**self.initial_state)

        return True

    # SENSES
    def get_proprioception(self):
        """Get proprioception sensor readings."""
        robot_qpos, robot_qvel = self.get_robot_state()
        proprioception = np.concatenate([robot_qpos, robot_qvel])

        return proprioception

    def get_touch(self):
        """Get touch sensor readings."""
        if self.touch:
            touch = self.data.sensordata[self._touch_sensor_id]
        else:
            touch = None

        return touch

    def get_vision(self):
        """Get vision sensor readings."""
        if self.vision:
            tmp_render_mode = self.render_mode
            self.render_mode = "rgb_array"
            vision_input = self.render()
            self.render_mode = tmp_render_mode
        else:
            vision_input = None

        return vision_input

    def _get_obs(self):
        """Get proprioception, touch and vision sensor readings."""
        return {
            'observation': Sensation(
                proprioception=self.get_proprioception(),
                touch=self.get_touch(),
                vision=self.get_vision(),
                goal=self.goal.copy()
            ),
        }

    def _get_info(self):
        """Get dict containing additional information about the current state of the environment."""
        return {
            "auxiliary_performances": {}
        }

    # ABSTRACT METHODS
    @abc.abstractmethod
    def _sample_goal(self) -> np.ndarray:
        raise NotImplementedError("Must be implemented in subclass.")

    def _render_callback(self, **kwargs):
        """A callback that is called before rendering. Can be used to implement custom rendering."""
        pass

    def set_initial_qpos(self, initial_qpos):
        """Set the initial qpos of the simulation based on given configuration. If the configuration is only specific
        to the robot, the rest of the qpos is left unchanged."""

        if initial_qpos is not None and len(initial_qpos) == 24:
            final_initial_qpos = self.data.qpos.copy()
            final_initial_qpos[0:24] = mj_qpos_dict_to_qpos_vector(self.model, initial_qpos)

            return final_initial_qpos
        elif initial_qpos is None or not initial_qpos:
            initial_qpos = self.data.qpos[:]

        return initial_qpos

    # MODEL PROPERTIES
    @property
    def actuator_names(self):
        return mj_get_category_names(self.model, "actuator")[0]

    @property
    def joint_names(self):
        return mj_get_category_names(self.model, "jnt")[0]
