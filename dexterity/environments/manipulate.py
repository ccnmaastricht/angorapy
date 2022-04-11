import os

import mujoco_py
import numpy as np
from gym import utils
from gym.envs.robotics import rotations
from gym.envs.robotics.utils import robot_get_obs
from scipy.spatial import transform

from dexterity.common.const import VISION_WH, N_SUBSTEPS
from dexterity.common.reward import manipulate
from dexterity.common.senses import Sensation
from dexterity.configs.reward_config import MANIPULATE_BASE
from dexterity.environments.shadowhand import BaseShadowHandEnv, get_palm_position, MODEL_PATH_MANIPULATE, FINGERTIP_SITE_NAMES

MANIPULATE_BLOCK_XML = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))),
                                    "assets/hand",
                                    'manipulate_block_touch_sensors.xml')
MANIPULATE_EGG_XML = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))),
                                  "assets/hand",
                                  'manipulate_egg_touch_sensors.xml')


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


class BaseManipulate(BaseShadowHandEnv):
    """Base class for in-hand object manipulation tasks."""

    max_steps_per_goal = 100

    def __init__(
            self,
            model_path,
            target_position,
            target_rotation,
            target_position_range,
            initial_qpos=None,
            randomize_initial_position=True,
            randomize_initial_rotation=True,
            distance_threshold=0.01,
            rotation_threshold=0.4,
            relative_control=False,
            ignore_z_target_rotation=False,
            n_substeps=N_SUBSTEPS,
            delta_t=0.002,
            touch_get_obs="sensordata",
            vision=False,
            touch=True
    ):
        """Initializes a new Hand manipulation environment.

        Args:
            model_path (string): path to the environments XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation
            vision (bool): indicator whether the environment should return frames (True) or the exact object
                position (False)
        """
        if vision:
            # init rendering [IMPORTANT]
            from mujoco_py import GlfwContext
            GlfwContext(offscreen=True)  # in newer version of gym use quiet=True to silence this

        self.touch_get_obs = touch_get_obs
        self.vision = vision
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]

        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.parallel_quats = [rotations.euler2quat(r) for r in rotations.get_parallel_rotations()]
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.ignore_z_target_rotation = ignore_z_target_rotation

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']
        initial_qpos = initial_qpos or {}

        super().__init__(initial_qpos=initial_qpos,
                         distance_threshold=0.1,
                         n_substeps=n_substeps,
                         delta_t=delta_t,
                         relative_control=relative_control,
                         model=model_path)

        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []

        # get touch sensor site names and their ids
        for k, v in self.sim.model._sensor_name2id.items():
            if 'robot0:TS_' in k:
                self._touch_sensor_id_site_id.append(
                    (v, self.sim.model._site_name2id[k.replace('robot0:TS_', 'robot0:T_')]))
                self._touch_sensor_id.append(v)

        # set touch sensors rgba values
        for _, site_id in self._touch_sensor_id_site_id:
            self.sim.model.site_rgba[site_id][3] = 0.0

        self.consecutive_goals_reached = 0
        self.steps_with_current_goal = 0
        self.previous_achieved_goal = self._get_achieved_goal()
        self._set_default_reward_function_and_config()

    def _set_default_reward_function_and_config(self):
        self.reward_function = manipulate
        self.reward_config = MANIPULATE_BASE

    def assert_reward_setup(self):
        """Assert whether the reward config fits the environment. """
        assert set(MANIPULATE_BASE.keys()).issubset(
            self.reward_config.keys()), "Incomplete manipulate reward configuration."

    def _get_achieved_goal(self):
        # Object position and rotation.
        object_qpos = self.sim.data.get_joint_qpos('object:joint')
        assert object_qpos.shape == (7,)
        return object_qpos.copy()

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])

        if self.target_position != 'ignore':
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff

        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot

        return achieved_both

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def reset(self):
        self.consecutive_goals_reached = 0
        self.steps_with_current_goal = 0

        return super().reset()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == 'z':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('object:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            target_quat = self.sim.data.get_joint_qpos('object:joint')
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self, render_targets=False):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.

        if render_targets:
            goal = self.goal.copy()
            assert goal.shape == (7,)
            if self.target_position == 'ignore':
                # Move the object to the side since we do not care about it's position.
                goal[0] += 0.15
            self.sim.data.set_joint_qpos('target:joint', goal)
            self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

            if 'object_hidden' in self.sim.model.geom_names:
                hidden_id = self.sim.model.geom_name2id('object_hidden')
                self.sim.model.geom_rgba[hidden_id, 3] = 1.

        self.sim.forward()

    def _get_obs(self):
        object_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        target_orientation = self.goal.ravel().copy()[3:]
        hand_joint_angles, hand_joint_velocities = robot_get_obs(self.sim)

        proprioception = np.concatenate([
            hand_joint_angles,
            hand_joint_velocities,
            object_qpos,
        ])

        return {
            "observation": Sensation(
                vision=None,
                proprioception=proprioception.copy(),
                somatosensation=None,
                goal=target_orientation),
            "achieved_goal": object_qpos.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

    def _is_dropped(self) -> bool:
        """Heuristically determine whether the object still is in the hand."""

        # determine object center position
        obj_center_idx = self.sim.model.site_name2id('object:center')
        obj_center_pos = self.sim.data.site_xpos[obj_center_idx]

        # determine palm center position
        palm_center_pos = get_palm_position(self.sim)

        dropped = (
                obj_center_pos[2] < palm_center_pos[2]  # z axis of object smaller than that of palm
            # we could add smth like checking for contacts between palm and object here, but the above works
            # pretty well already tbh
        )

        return dropped

    def _goal_progress(self):
        return (sum(self._goal_distance(self.goal, self.previous_achieved_goal))
                - sum(self._goal_distance(self.goal, self._get_achieved_goal())))

    def step(self, action):
        """Make step in environment."""
        obs, reward, done, info = super().step(action)
        self.steps_with_current_goal += 1

        success = self._is_success(self._get_achieved_goal(), self.goal)

        # determine if a goal has been reached
        if not success:
            self.previous_achieved_goal = self._get_achieved_goal().copy()
        else:
            self.consecutive_goals_reached += 1
            self.goal = self._sample_goal()
            self.steps_with_current_goal = 0
            obs = self._get_obs()

        if self.steps_with_current_goal >= BaseManipulate.max_steps_per_goal:
            done = True

        # determine if done
        dropped = self._is_dropped()
        done = done or dropped or self.consecutive_goals_reached >= 50

        return obs, reward, done, info


class ManipulateBlock(BaseManipulate, utils.EzPickle):
    """Manipulate Environment with a Block as an object."""

    def __init__(self,
                 target_position='ignore',
                 target_rotation='xyz',
                 touch_get_obs='sensordata',
                 relative_control=True,
                 vision: bool = False,
                 delta_t: float = 0.002):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, "dense")
        BaseManipulate.__init__(self,
                                model_path=MODEL_PATH_MANIPULATE,
                                touch_get_obs=touch_get_obs,
                                target_rotation=target_rotation,
                                target_position=target_position,
                                target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                                vision=vision,
                                relative_control=relative_control,
                                delta_t=delta_t
                                )


class OpenAIManipulate(BaseManipulate, utils.EzPickle):

    def __init__(self, delta_t=0.002):
        utils.EzPickle.__init__(self, "ignore", "xyz", 'sensordata', "dense")
        BaseManipulate.__init__(self,
                                model_path=MODEL_PATH_MANIPULATE,
                                touch_get_obs='sensordata',
                                target_rotation="xyz",
                                target_position="ignore",
                                target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                                vision=False,
                                n_substeps=10,
                                delta_t=delta_t,
                                relative_control=True
                                )

    def _get_obs(self):
        finger_tip_positions = np.array([self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]).flatten()
        object_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        target_orientation = self.goal.ravel().copy()[3:]

        target_quat = transform.Rotation.from_quat(target_orientation)
        current_quat = transform.Rotation.from_quat((object_qpos[3:]))
        relative_target_orientation = (target_quat * current_quat.inv()).as_quat()

        hand_joint_angles, hand_joint_velocities = robot_get_obs(self.sim)
        object_positional_velocity = self.sim.data.get_body_xvelp('object')
        object_angular_velocity = self.sim.data.get_body_xvelr('object')  # quaternions in OpenAI model

        proprioception = np.concatenate([
            finger_tip_positions,
            object_qpos,
            relative_target_orientation,
            hand_joint_angles,
            hand_joint_velocities,
            object_positional_velocity,
            object_angular_velocity
        ])

        return {
            "observation": Sensation(
                vision=None,
                proprioception=proprioception.copy(),
                somatosensation=None,
                goal=target_orientation),
            "achieved_goal": object_qpos.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


class HumanoidManipulateBlockDiscrete(ManipulateBlock):
    continuous = False
    asynchronous = False

    def _get_obs(self):
        """Gather humanoid senses and asynchronous information."""

        # vision
        object_qpos = self.sim.data.get_joint_qpos('object:joint').copy()

        # goal
        target_orientation = self.goal.ravel().copy()[3:]

        # proprioception
        hand_joint_angles, hand_joint_velocities = robot_get_obs(self.sim)

        proprioception = np.concatenate([
            hand_joint_angles,
            hand_joint_velocities
        ])

        # touch
        touch = self.sim.data.sensordata[self._touch_sensor_id]

        # asynchronous information
        finger_tip_positions = np.array([self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]).flatten()
        target_quat = transform.Rotation.from_quat(target_orientation)
        current_quat = transform.Rotation.from_quat((object_qpos[3:]))
        relative_target_orientation = (target_quat * current_quat.inv()).as_quat()

        object_positional_velocity = self.sim.data.get_body_xvelp('object')
        object_angular_velocity = self.sim.data.get_body_xvelr('object')  # quaternions in OpenAI model

        asynchronous = np.concatenate([
            finger_tip_positions,
            relative_target_orientation,
            object_positional_velocity,
            object_angular_velocity
        ])

        return {
            "observation": Sensation(
                vision=object_qpos,
                proprioception=proprioception.copy(),
                somatosensation=touch,
                goal=target_orientation,
                asynchronous=None if not self.asynchronous else asynchronous
            ),
            "achieved_goal": object_qpos.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


class HumanoidManipulateBlockDiscreteAsynchronous(HumanoidManipulateBlockDiscrete):
    asynchronous = True


class OpenAIManipulateDiscrete(OpenAIManipulate):
    continuous = False


class ManipulateBlockDiscrete(ManipulateBlock):
    continuous = False


class ManipulateEgg(BaseManipulate, utils.EzPickle):
    """Manipulate Environment with an Egg as an object."""

    def __init__(self, target_position='ignore', target_rotation='xyz', touch_get_obs='sensordata',
                 relative_control=True,
                 vision: bool = False):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, "dense")
        BaseManipulate.__init__(self,
                                model_path=MANIPULATE_EGG_XML,
                                touch_get_obs=touch_get_obs,
                                target_rotation=target_rotation,
                                target_position=target_position,
                                target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                                vision=vision,
                                relative_control=relative_control
                                )
