from typing import List
from typing import Optional

import dm_control
import mujoco
import numpy as np
from dm_control.utils import transformations
from gymnasium import utils
from scipy.spatial import transform

import angorapy.tasks.utils
from angorapy.common.const import N_SUBSTEPS
from angorapy.common.const import VISION_WH
from angorapy.common.senses import Sensation
from angorapy.tasks.envs.dexterity.consts import DEFAULT_INITIAL_QPOS
from angorapy.tasks.envs.dexterity.consts import FINGERTIP_SITE_NAMES
from angorapy.tasks.envs.dexterity.core import BaseShadowHandEnv
from angorapy.tasks.envs.dexterity.mujoco_model.worlds.manipulation import ShadowHandWithCubeWorld
from angorapy.tasks.envs.dexterity.reward import manipulate
from angorapy.tasks.envs.dexterity.reward_configs import MANIPULATE_BASE
from angorapy.tasks.utils import quat_mul, quat_conjugate
from angorapy.utilities.math import quaternions

chain_code_dict = {
    "r": quaternions.from_angle_and_axis(np.pi / 2, np.array([1., 0., 0.])),
    "l": quaternions.from_angle_and_axis(-np.pi / 2, np.array([1., 0., 0.])),
    "u": quaternions.from_angle_and_axis(np.pi / 2, np.array([0., 1., 0.])),
    "d": quaternions.from_angle_and_axis(-np.pi / 2, np.array([0., 1., 0.])),
    "c": quaternions.from_angle_and_axis(np.pi / 2, np.array([0., 0., 1.])),
    "a": quaternions.from_angle_and_axis(-np.pi / 2, np.array([0., 0., 1.])),
}


def calc_rotation_chain(chain_code: str, current_rotation: np.ndarray) -> List[np.ndarray]:
    """
    From a given base rotation, calculate a chain of rotations based on a given code.

    Code elements are as follows:
        - l: rotate left
        - r: rotate right
        - u: rotate up
        - d: rotate down
        - c: rotate clockwise
        - a: rotate anti-clockwise

    Steps are segmented by _ underscores. For example, the code "l_r" will rotate the object left and then right.
    The code "luc_rd" will first rotate the object left, up, clockwise, and second rotate it right and down.

    Args:
        chain_code:
        current_rotation:

    Returns:
        the list of rotations (quaternions) represented by the chain code
    """
    rotations = []

    for step in chain_code.split("_"):
        for code in step:
            current_rotation = quaternions.multiply(chain_code_dict[code], current_rotation)

        rotations.append(current_rotation.copy())

    return rotations


def calc_rotation_set(current_rotation):
    """From a given base rotation, calculate all 24 possible rotations of an object in steps of 90 degrees."""

    # get rotations for all 24 possible orientations
    deg90 = np.pi / 2
    deg180 = np.pi
    x_axis = np.array([1., 0., 0.])
    y_axis = np.array([0., 1., 0.])
    z_axis = np.array([0., 0., 1.])

    base_up_faces = [
        current_rotation,
        quat_mul(current_rotation, quaternions.from_angle_and_axis(deg90, x_axis)),
        quat_mul(current_rotation, quaternions.from_angle_and_axis(-deg90, x_axis)),
        quat_mul(current_rotation, quaternions.from_angle_and_axis(deg90, y_axis)),
        quat_mul(current_rotation, quaternions.from_angle_and_axis(-deg90, y_axis)),
        quat_mul(current_rotation, quaternions.from_angle_and_axis(deg180, y_axis)),
    ]

    test_cases_block_rotations = []
    test_cases_block_rotations += base_up_faces
    for base_up_face in base_up_faces:
        test_cases_block_rotations.append(
            quat_mul(base_up_face, quaternions.from_angle_and_axis(deg90, z_axis))
        )
        test_cases_block_rotations.append(
            quat_mul(base_up_face, quaternions.from_angle_and_axis(-deg90, z_axis))
        )
        test_cases_block_rotations.append(
            quat_mul(base_up_face, quaternions.from_angle_and_axis(deg180, z_axis))
        )

    return test_cases_block_rotations


class BaseManipulate(BaseShadowHandEnv):
    """Base class for in-hand object manipulation tasks."""

    max_steps_per_goal = 100

    def __init__(
            self,
            target_position,
            target_rotation,
            target_position_range,
            initial_qpos=DEFAULT_INITIAL_QPOS,
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
            touch=True,
            render_mode: Optional[str] = None
    ):
        """Initializes a new Hand manipulation environment.

        Args:
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
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered
            achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered
            achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative
            to the current state
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation
            vision (bool): indicator whether the environment should return frames (True) or the exact object
                position (False)
        """

        self.world = ShadowHandWithCubeWorld()
        self.object_id = self.world.cube.name + "/"
        self.object_joint_id = f'{self.object_id}object:joint/'
        self.object_center_id = f'{self.object_id}object:center'

        self.touch_get_obs = touch_get_obs
        self.vision = vision
        self.touch = touch
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]

        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.parallel_quats = [angorapy.tasks.utils.euler2quat(r) for r in
                               angorapy.tasks.utils.get_parallel_rotations()]
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.ignore_z_target_rotation = ignore_z_target_rotation

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']

        self.consecutive_goals_reached = 0
        self.steps_with_current_goal = 0
        self.previous_object_pose = self.get_object_pose()

        super().__init__(
            initial_qpos=initial_qpos,
            n_substeps=n_substeps,
            delta_t=delta_t,
            relative_control=relative_control,
            model=self.world.root,
            vision=vision,
            touch=touch,
            render_mode=render_mode
        )

        # set touch sensors rgba values
        for _, site_id in self._touch_sensor_id_site_id:
            self.model.site_rgba[site_id][3] = 0.0

        # determine the face up rotations of the cube
        x_axis = np.array([1., 0., 0.])
        y_axis = np.array([0., 1., 0.])
        z_axis = np.array([0., 0., 1.])
        deg90 = np.pi / 2
        deg180 = np.pi

        self.FACE_UP_ROTATIONS = [
            quaternions.from_angle_and_axis(deg90, x_axis),
            quaternions.from_angle_and_axis(-deg90, x_axis),
            quaternions.from_angle_and_axis(deg90, y_axis),
            quaternions.from_angle_and_axis(-deg90, y_axis),
            quaternions.from_angle_and_axis(deg180, y_axis),
            quaternions.from_angle_and_axis(-deg180, x_axis),
        ]

    def _set_default_reward_function_and_config(self):
        self.reward_function = manipulate
        self.reward_config = MANIPULATE_BASE

    def assert_reward_setup(self):
        """Assert whether the reward config fits the environment. """
        assert set(MANIPULATE_BASE.keys()).issubset(
            self.reward_config.keys()), "Incomplete manipulate reward configuration."

    def get_object_pose(self):
        """Object position and rotation."""
        if hasattr(self, "data"):
            object_qpos = self.data.jnt(self.object_joint_id).qpos
            assert object_qpos.shape == (7,)
            return object_qpos.copy()
        else:
            # simulation not initialized yet
            return np.zeros(7)  # todo double check

    def _goal_distance(self,
                       goal_a,
                       goal_b):
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
                euler_a = angorapy.tasks.utils.quat2euler(quat_a)
                euler_b = angorapy.tasks.utils.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = angorapy.tasks.utils.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = quat_mul(quat_a, angorapy.tasks.utils.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff

        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    def _is_success(self,
                    achieved_goal,
                    desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot

        return achieved_both

    def _env_setup(self,
                   initial_state):
        super()._env_setup(initial_state)

        self.initial_goal = self.get_object_pose().copy()
        self.goal = self._sample_goal()

    def reset(self,
              **kwargs):
        self.consecutive_goals_reached = 0
        self.steps_with_current_goal = 0

        return super().reset(**kwargs)

    def _reset_sim(self):
        self.reset_model()

        initial_qpos = self.data.jnt(self.object_joint_id).qpos.copy()
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
                offset_quat = quaternions.from_angle_and_axis(angle, axis)
                initial_quat = quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quaternions.from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = quat_mul(z_quat, parallel_quat)
                initial_quat = quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quaternions.from_angle_and_axis(angle, axis)
                initial_quat = quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise ValueError('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.data.jnt(self.object_joint_id).qpos[:] = initial_qpos

        def is_on_palm():
            mujoco.mj_forward(self.model, self.data)
            cube_middle_pos = self.data.site(self.object_center_id).xpos
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(self.action_space.shape[0]))
            mujoco.mj_step(self.model, self.data)

        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_pos = self.data.jnt(self.object_joint_id).qpos[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            target_pos = self.data.jnt(self.object_joint_id).qpos[:3]
        else:
            raise ValueError('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quaternions.from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quaternions.from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = quaternions.from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            target_quat = self.data.jnt(self.object_joint_id).qpos
        else:
            raise ValueError('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self,
                         render_targets=False):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.

        if render_targets:
            goal = self.goal.copy()
            assert goal.shape == (7,)
            if self.target_position == 'ignore':
                # Move the object to the side since we do not care about it's position.
                goal[0] += 0.15

            self.data.jnt('target:joint').qpos[:] = goal
            self.data.jnt('target:joint').qvel[:] = np.zeros(6)

            if 'object_hidden' in self.model.names.split(b"\x00"):
                hidden_id = self.model.geom_name2id('object_hidden')
                self.model.geom_rgba[hidden_id, 3] = 1.

        mujoco.mj_forward(self.model, self.data)

    def get_vision(self):
        if not self.vision:
            object_qpos = self.data.jnt(self.object_joint_id).qpos.copy()
            vision = object_qpos.astype(np.float32)
        else:
            vision = super().get_vision()

        return vision

    def _get_obs(self):
        """Gather humanoid senses and asymmetric information."""
        object_qpos = self.get_object_pose()
        vision_input = self.get_vision()
        target_orientation = self.goal.ravel().copy()[3:]
        proprioception = self.get_proprioception()
        touch = self.get_touch()

        # asymmetric information
        finger_tip_positions = np.array([self.data.site(name).xpos for name in FINGERTIP_SITE_NAMES]).flatten()
        target_quat = transform.Rotation.from_quat(target_orientation)
        current_quat = transform.Rotation.from_quat((object_qpos[3:]))
        relative_target_orientation = (target_quat * current_quat.inv()).as_quat()

        object_positional_velocity = self.data.body(self.object_id).cvel[3:]  # todo double check if correct part
        object_angular_velocity = self.data.body(self.object_id).cvel[:3]  # quaternions in OpenAI model

        asymmetric = np.concatenate([
            finger_tip_positions,
            relative_target_orientation,
            object_positional_velocity,
            object_angular_velocity
        ])

        return {
            "observation": Sensation(
                vision=vision_input,
                proprioception=proprioception.copy(),
                touch=touch,
                goal=target_orientation,
                asymmetric=None if not self.asymmetric else asymmetric
            ),
        }

    def _is_dropped(self) -> bool:
        """Heuristically determine whether the object still is in the hand."""

        # determine object center position
        obj_center_pos = self.data.site(self.object_center_id).xpos

        # determine palm center position
        palm_center_pos = self.data.site("robot/palm_center_site").xpos

        dropped = (
                obj_center_pos[2] < palm_center_pos[2]  # z axis of object smaller than that of palm
        )

        return dropped

    def _goal_progress(self):
        return (sum(self._goal_distance(self.goal, self.previous_object_pose))
                - sum(self._goal_distance(self.goal, self.get_object_pose())))

    def step(self,
             action):
        """Make step in environment."""
        obs, reward, terminated, truncated, info = super().step(action)
        self.steps_with_current_goal += 1

        success = self._is_success(self.get_object_pose(), self.goal)

        # determine if a goal has been reached
        if not success:
            self.previous_object_pose = self.get_object_pose().copy()
        else:
            self.consecutive_goals_reached += 1
            self.goal = self._sample_goal()
            self.steps_with_current_goal = 0
            obs = self._get_obs()

        if self.steps_with_current_goal >= BaseManipulate.max_steps_per_goal:
            terminated = True

        # determine if done
        dropped = self._is_dropped()
        terminated = terminated or dropped or self.consecutive_goals_reached >= 50

        if "auxiliary_performances" not in info.keys():
            info["auxiliary_performances"] = {}
        info["auxiliary_performances"]["consecutive_goals_reached"] = self.consecutive_goals_reached

        return obs, reward, terminated, truncated, info

    def _get_info(self):
        return {
            **super()._get_info(),
            "is_success": self._is_success(self.goal, self.get_object_pose()),
            "current_face_up": self.get_current_face_up(),
            "goal_distance": self._goal_distance(self.goal, self.get_object_pose()),
            "achieved_goal": self.get_object_pose().copy(),
            "desired_goal": self.goal.copy(),
        }

    def get_current_face_up(self):
        current_pose = self.get_object_pose()

        angle_diffs = []
        quat_a = current_pose[3:]
        for i, face_up_rotation in enumerate(self.FACE_UP_ROTATIONS):
            quat_diff = quat_mul(quat_a, angorapy.tasks.utils.quat_conjugate(face_up_rotation))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            angle_diffs.append(angle_diff)

        return np.argmin(angle_diffs)


class ManipulateBlock(BaseManipulate, utils.EzPickle):
    """Manipulate Environment with a Block as an object."""

    asymmetric = False
    continuous = True

    def __init__(self,
                 target_position='ignore',
                 target_rotation='xyz',
                 touch_get_obs='sensordata',
                 relative_control=True,
                 vision: bool = False,
                 delta_t: float = 0.002,
                 render_mode: Optional[str] = None):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, "dense")
        BaseManipulate.__init__(
            self,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            vision=vision,
            relative_control=relative_control,
            delta_t=delta_t,
            render_mode=render_mode
        )


class ManipulateBlockDiscrete(ManipulateBlock):
    asymmetric = False
    continuous = False


class ManipulateBlockDiscreteAsymmetric(ManipulateBlockDiscrete):
    asymmetric = True
    continuous = False


class ManipulateBlockAsymmetric(ManipulateBlockDiscrete):
    asymmetric = True
    continuous = True


class NoisyManipulateBlock(ManipulateBlock):
    """Manipulate Environment with a Block as an object."""

    asymmetric = True
    continuous = False

    ROTATION_NOISE = np.pi / 36
    POSITION_NOISE = 3.71

    def __init__(self, *args, **kwargs):
        self.noisy_rotation = True
        self.noisy_position = True
        self.not_yet_warned = True

        super().__init__(*args, **kwargs)

    def toggle_rotation_noise(self):
        self.noisy_rotation = not self.noisy_rotation

    def toggle_position_noise(self):
        self.noisy_position = not self.noisy_position

    def get_vision(self):
        """Get (surrogate) vision with added noise."""

        if not self.vision:
            object_qpos = self.data.jnt(self.object_joint_id).qpos.copy()
            vision = np.copy(object_qpos.astype(np.float32))

            if self.noisy_rotation:
                # get random quaternion rotating by max 5 degrees in total
                random_total_angle = np.random.normal(0, self.ROTATION_NOISE)
                angle_split = np.random.uniform(0, 1, 3)
                angle_split /= angle_split.sum()
                angle_noise_by_axis = angle_split * random_total_angle

                x_rotation_noise_quaternion = quaternions.from_angle_and_axis(angle_noise_by_axis[0], np.array([1., 0., 0.]))
                y_rotation_noise_quaternion = quaternions.from_angle_and_axis(angle_noise_by_axis[1], np.array([0., 1., 0.]))
                z_rotation_noise_quaternion = quaternions.from_angle_and_axis(angle_noise_by_axis[2], np.array([0., 0., 1.]))

                rotation_noise_quaternion = quat_mul(x_rotation_noise_quaternion,
                                                                          y_rotation_noise_quaternion)
                rotation_noise_quaternion = quat_mul(rotation_noise_quaternion,
                                                                          z_rotation_noise_quaternion)

                vision[3:] = quat_mul(vision[3:], rotation_noise_quaternion)
            else:
                if self.not_yet_warned:
                    print("WARNING: No rotation noise added to vision.")
                    self.not_yet_warned = False

            if self.noisy_position:
                random_displacement_vector = np.random.normal(size=3)
                random_displacement_vector /= np.linalg.norm(random_displacement_vector)

                random_displacement_vector *= (self.POSITION_NOISE / 1000)

                vision[:3] += random_displacement_vector
            else:
                if self.not_yet_warned:
                    print("WARNING: No position noise added to vision.")
                    self.not_yet_warned = False
        else:
            vision = super().get_vision()

        return vision

    def _get_obs(self):
        obs = super()._get_obs()

        asymmetric = obs["observation"].asymmetric
        asymmetric = np.concatenate([
            asymmetric,
            self.get_object_pose()
        ], dtype=np.float32)

        obs["observation"].asymmetric = asymmetric

        return obs


class TripleCamManipulateBlock(NoisyManipulateBlock):
    asymmetric = True
    continuous = False

    def __init__(self, *args, **kwargs):
        self.vision = True
        self.cameras = []
        super().__init__(*args, **kwargs)

    def _env_setup(self, *args, **kwargs):
        super()._env_setup(*args, **kwargs)

        self.renderer = mujoco.Renderer(self.model, height=VISION_WH, width=VISION_WH)
        self.cameras = [self._get_viewer("rgb_array").cam]
        for i in range(1, 3):
            self.cameras.append(mujoco.MjvCamera())
            self.cameras[-1].type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cameras[-1].fixedcamid = -1

            self.cameras[-1].distance = self.cameras[0].distance
            self.cameras[-1].lookat[:] = self.cameras[0].lookat[:]
            self.cameras[-1].elevation = self.cameras[0].elevation
            self.cameras[-1].azimuth = self.cameras[0].azimuth

            self.cameras[-1].azimuth += [35, -35][i - 1]  # wrist to the bottom
            self.cameras[-1].elevation += 45

            self.cameras[-1].distance -= 0.1  # wrist to the bottom

    def get_vision(self):
        if self.vision:
            vision = []
            for camera in self.cameras:
                self.renderer.update_scene(self.data, camera)
                image = self.renderer.render()
                vision.append(image)

            vision = np.concatenate(vision, axis=-1)
        else:
            vision = super().get_vision()

        return vision


class TestCaseManipulateBlock(ManipulateBlock):
    asymmetric = True
    continuous = False

    def __init__(
            self,
            target_position='ignore',
            target_rotation='xyz',
            touch_get_obs='sensordata',
            relative_control=True,
            vision: bool = False,
            delta_t: float = 0.002,
            render_mode: Optional[str] = None
    ):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, "dense")

        self.chain_code = "l_r_u_d_c_a"
        self.position_in_chain = -1
        self.target_chain = [np.array([1., 0., 0., 0.])]
        self.chain_done = False

        BaseManipulate.__init__(
            self,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            vision=vision,
            relative_control=relative_control,
            delta_t=delta_t,
            render_mode=render_mode,
            randomize_initial_rotation="parallel",
        )

        self.test_cases_block_rotations = calc_rotation_set(self.get_object_pose()[3:])

        self.set_chain_code(self.chain_code)

    def set_chain_code(self, chain_code: str):
        assert set(chain_code).issubset(set(list(chain_code_dict.keys()) + ["_"])), \
            "Chain code contains invalid characters."

        self.chain_code = chain_code
        self.target_chain = calc_rotation_chain(chain_code=self.chain_code,
                                                current_rotation=self.get_object_pose()[3:])
        self.chain_done = False

    def _sample_goal(self):
        self.position_in_chain += 1

        if self.position_in_chain >= len(self.target_chain):
            self.chain_done = True
            self.position_in_chain = -1

        return np.concatenate([self.get_object_pose()[:3], self.target_chain[self.position_in_chain]])

    def _get_info(self):
        return {
            **super()._get_info(),
            "chain_code": self.chain_code,
            "position_in_chain": self.position_in_chain,
            "current_intention": self.chain_code.split("_")[self.position_in_chain] if self.position_in_chain != -1 else "done",
        }

    def step(self, action):
        """Make step in environment."""
        obs, reward, terminated, truncated, info = super().step(action)

        if self.chain_done:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.position_in_chain = -1
        self.set_chain_code(self.chain_code)
        self.chain_done = False

        return super().reset(**kwargs)

    def _reset_sim(self):
        self.reset_model()

        initial_qpos = self.data.jnt(self.object_joint_id).qpos.copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            angle = self.np_random.uniform(-np.pi / 15, np.pi / 15)
            axis = self.np_random.uniform(-1., 1., size=3)
            offset_quat = quaternions.from_angle_and_axis(angle, axis)
            initial_quat = quat_mul(initial_quat, offset_quat)

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.003)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.data.jnt(self.object_joint_id).qpos[:] = initial_qpos

        def is_on_palm():
            mujoco.mj_forward(self.model, self.data)
            cube_middle_pos = self.data.site(self.object_center_id).xpos
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(self.action_space.shape[0]))
            mujoco.mj_step(self.model, self.data)

        return is_on_palm()