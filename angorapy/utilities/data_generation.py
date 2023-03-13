"""Generation of auxiliary datasets."""
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from angorapy.common.const import VISION_WH
from angorapy.common.loss import multi_point_euclidean_distance
from angorapy.common.wrappers import make_env
from angorapy.environments import *


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

        return tf.reshape(parsed["image"], (VISION_WH, VISION_WH, 3)), tf.reshape(parsed["pose"], (7,))

    return tf.data.TFRecordDataset(path).map(_parse_function)


def gen_cube_quats_prediction_data(n: int, save_path: str):
    """Generate dataset of hand images with cubes as data points and the pos and rotation of the cube as targets."""
    hand_env = gym.make("HumanoidVisualManipulateBlock-v0", render_mode="rgb_array")

    # sample = hand_env.observation_space.sample()
    # X, Y = np.empty((n,) + sample["observation"]["vision"].shape, dtype=np.float16), \
    #     np.empty((n,) + sample["achieved_goal"].shape, dtype=np.float16)

    hand_env.reset()

    os.makedirs("storage/data/pretraining", exist_ok=True)
    with tf.io.TFRecordWriter(save_path) as writer:
        for i in tqdm(range(n), desc="Sampling Data"):
            sample, r, terminated, truncated, info = hand_env.step(hand_env.action_space.sample())
            done = terminated or truncated
            image = sample["observation"]["vision"] / 255
            quaternion = sample["achieved_goal"]
            # X[i] = image
            # Y[i] = quaternion

            example = serialize_example(tf.io.serialize_tensor(image), tf.io.serialize_tensor(quaternion))
            writer.write(example)

            if done or i % 64 == 0:
                hand_env.reset()

    return load_dataset(save_path)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    data = gen_cube_quats_prediction_data(8000, "../storage/data/pretraining/pose_data.tfrecord")
    print(data)