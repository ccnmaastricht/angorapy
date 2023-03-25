"""Generation of auxiliary datasets."""
import os
from typing import Tuple

import mujoco
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from angorapy.agent import PPOAgent
from angorapy.common.const import VISION_WH
from angorapy.common.loss import multi_point_euclidean_distance
from angorapy.common.wrappers import make_env
from angorapy.environments import *
from angorapy.environments.anthrobotics import AnthropomorphicEnv
import tensorflow_graphics.geometry.transformation as tfg


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, pose):
    feature = {
      'image': _bytes_feature(image),
      'pose': _bytes_feature(pose),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def load_dataset(path):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "pose": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        parsed["image"] = tf.io.parse_tensor(parsed["image"], out_type=tf.float32)
        parsed["pose"] = tf.io.parse_tensor(parsed["pose"], out_type=tf.float32)

        return tf.reshape(parsed["image"], (VISION_WH, VISION_WH, -1)), tf.reshape(parsed["pose"], (7,))

    return tf.data.TFRecordDataset(path).map(_parse_function)


def gen_cube_quats_prediction_data(n: int, save_path: str, binocular=False, n_cameras: int = 1):
    """Generate dataset of hand images with cubes as data points and the pos and rotation of the cube as targets."""
    agent = PPOAgent.from_agent_state(1670596186987840, "best", path_modifier="../")
    agent.policy, agent.value, agent.joint = agent.build_models(agent.joint.get_weights(), batch_size=1, sequence_length=1)

    hand_env = make_env("HumanoidVisualManipulateBlockDiscreteAsynchronous-v0",
                        render_mode="rgb_array",
                        transformers=agent.env.transformers)

    env_unwrapped: AnthropomorphicEnv = hand_env.unwrapped
    model = env_unwrapped.model
    data = env_unwrapped.data
    renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)
    cameras = [env_unwrapped._get_viewer("rgb_array").cam]
    for i in range(1, n_cameras):
        cameras.append(mujoco.MjvCamera())
        cameras[-1].type = mujoco.mjtCamera.mjCAMERA_FREE
        cameras[-1].fixedcamid = -1
        for j in range(3):
            cameras[-1].lookat[j] = np.median(data.geom_xpos[:, j])
        cameras[-1].distance = model.stat.extent

        cameras[-1].distance = 0.3  # zoom in
        cameras[-1].azimuth = -90.0 + [20, -20][i - 1]  # wrist to the bottom
        cameras[-1].elevation = -90.0 + [20, -20][i - 1]  # top down view
        cameras[-1].lookat[1] += 0.03  # slightly move forward

    def random_augmentation():
        # Choose a random number between 0 and 1
        p = np.random.random()

        # Get the quaternion for the current orientation of the body
        object = data.jnt("object:joint")
        quat = data.jnt("object:joint").qpos[3:]
        original_qpos = object.qpos.copy()

        if p < 0.0:  # Leave the object pose as is
            pass
        elif p < 1:  # Rotate the object by 90 degrees around its main axes
            # define the rotation
            axis = np.random.choice(['x', 'y', 'z'])
            sign = np.random.choice([-1, 1])

            # rotate the body by 90 degrees around the chosen axis
            if axis == 'x':
                rotation_quat = tfg.quaternion.from_axis_angle(np.array([[1., 0., 0.]]), np.array([[sign * np.pi / 2]]))
            elif axis == 'y':
                rotation_quat = tfg.quaternion.from_axis_angle(np.array([[0., 1., 0.]]), np.array([[sign * np.pi / 2]]))
            else:
                rotation_quat = tfg.quaternion.from_axis_angle(np.array([[0., 0., 1.]]), np.array([[sign * np.pi / 2]]))

            # Apply the rotation to the body quaternion
            new_quat = tfg.quaternion.multiply(quat, rotation_quat)
            object.qpos[3:] = new_quat
        else:  # "Jitter" the object by adding Gaussian noise to position and orientation
            # Add Gaussian noise to the body position
            noise_pos = np.random.normal(loc=0, scale=0.01, size=3)
            object.qpos[:3] += noise_pos

            # Add Gaussian noise to the body orientation
            noise_quat = np.random.normal(loc=0, scale=0.01, size=4)
            noise_quat /= np.linalg.norm(noise_quat)
            new_quat = tfg.quaternion.multiply(quat, noise_quat)
            data.qpos[3:] = new_quat

        mujoco.mj_forward(model, data)

        images = []
        for cam in cameras:
            renderer.update_scene(data, camera=cam)
            images.append(renderer.render().copy())

        qpos = object.qpos.copy()
        object.qpos = original_qpos

        # plt.imshow(np.concatenate(images, axis=-2))
        # plt.show()

        return tf.cast(tf.concat(images, axis=-1) / 255, dtype=tf.float32), tf.cast(qpos, tf.float32)

    os.makedirs("storage/data/pretraining", exist_ok=True)
    with tf.io.TFRecordWriter(save_path) as writer:
        state, _ = hand_env.reset()
        for i in tqdm(range(n), desc="Sampling Data"):
            state["vision"] = hand_env.data.jnt('object:joint').qpos.copy()

            action = agent.act(state)
            sample, r, terminated, truncated, info = hand_env.step(action)
            done = terminated or truncated
            image, quaternion = random_augmentation()
            # image, quaternion = sample["vision"], info["achieved_goal"]
            # X[i] = image
            # Y[i] = quaternion

            example = serialize_example(tf.io.serialize_tensor(image), tf.io.serialize_tensor(quaternion))
            writer.write(example)

            if done or i % 64 == 0:
                sample, _ = hand_env.reset()

            state = sample

    return load_dataset(save_path)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    data = gen_cube_quats_prediction_data(8000, "../storage/data/pretraining/pose_data.tfrecord")
    print(data)