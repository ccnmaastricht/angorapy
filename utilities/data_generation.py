"""Generation of auxiliary datasets."""
import os

import gym
from tqdm import tqdm

from environments import *
import numpy as np
import tensorflow as tf


def gen_cube_quats_prediction_data(n):
    """Generate dataset of hand images with cubes as data points and the pos and rotation of the cube as targets."""
    hand_env = gym.make("BaseShadowHand-v0")

    sample = hand_env.observation_space.sample()
    X, Y = np.empty((n,) + sample["observation"][0].shape), np.empty((n,) + sample["achieved_goal"].shape)

    hand_env.reset()
    for i in tqdm(range(n), desc="Sampling Data"):
        sample, r, done, info = hand_env.step(hand_env.action_space.sample())
        image = sample["observation"][0] / 255
        quaternion = sample["achieved_goal"]
        X[i] = image
        Y[i] = quaternion

        if done:
            hand_env.reset()

    return tf.convert_to_tensor(X.astype("float32")), tf.convert_to_tensor(Y.astype("float32"))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    data = gen_cube_quats_prediction_data(1000)
