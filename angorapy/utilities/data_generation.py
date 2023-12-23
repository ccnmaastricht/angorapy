"""Generation of auxiliary datasets."""
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from angorapy import make_task
from angorapy.agent import Agent
from angorapy.common.const import VISION_WH


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, pose):
    feature = {
        'sim_state': _bytes_feature(image),
        'pose': _bytes_feature(pose),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def load_unrendered_dataset(path):
    feature_description = {
        "sim_state": tf.io.FixedLenFeature([], tf.string),
        "pose": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        parsed["sim_state"] = tf.io.parse_tensor(parsed["sim_state"], out_type=tf.float32)
        parsed["pose"] = tf.io.parse_tensor(parsed["pose"], out_type=tf.float32)

        return tf.reshape(parsed["sim_state"], (61,)), tf.reshape(parsed["pose"], (7,))

    return tf.data.TFRecordDataset(path).map(_parse_function)


def gen_cube_quats_prediction_data(n: int, save_path: str):
    """Generate dataset of hand images with cubes as data points and the pos and rotation of the cube as targets."""
    import dexterity as dx

    agent = Agent.from_agent_state(1702645801966906, "best", path_modifier="")
    agent.policy, agent.value, agent.joint = agent.build_models(
        agent.joint.get_weights(),
        batch_size=1,
        sequence_length=1
    )

    hand_env = agent.env

    os.makedirs("storage/data/pretraining", exist_ok=True)
    with (tf.io.TFRecordWriter(save_path) as writer):
        state, info = hand_env.reset()
        for i in tqdm(range(n), desc="Sampling Data", disable=not agent.is_root):
            action = hand_env.action_space.sample()
            # agent.act(state)
            sample, r, terminated, truncated, info = hand_env.step(action)
            done = terminated or truncated

            pose = tf.cast(info["achieved_goal"], dtype=tf.float32)
            sim_state = hand_env.get_state()
            sim_state = tf.cast(np.concatenate([sim_state["qpos"], sim_state["qvel"]]), dtype=tf.float32)

            example = serialize_example(tf.io.serialize_tensor(sim_state), tf.io.serialize_tensor(pose))
            writer.write(example)

            if done:
                sample, _ = hand_env.reset()

            state = sample
