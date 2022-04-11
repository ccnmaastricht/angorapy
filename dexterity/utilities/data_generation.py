"""Generation of auxiliary datasets."""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dexterity.common.loss import multi_point_euclidean_distance
from dexterity.common.wrappers import make_env
from dexterity.environments import *


def gen_cube_quats_prediction_data(n):
    """Generate dataset of hand images with cubes as data points and the pos and rotation of the cube as targets."""
    hand_env = gym.make("BaseShadowHandEnv-v0")

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


def gen_hand_pos_prediction_data(n):
    """Generate dataset of hand images with cubes as data points and the pos and rotation of the cube as targets."""
    hand_env = make_env("ReachAbsoluteVisual-v0")

    sample = hand_env.observation_space.sample()
    X, Y = np.empty((n,) + sample["observation"]["vision"].shape), np.empty((n,) + sample["achieved_goal"].shape)

    hand_env.reset()
    for i in tqdm(range(n), desc="Sampling Data"):
        sample, r, done, info = hand_env.step(hand_env.action_space.sample() * 2)  # scale to have extremer actions
        image = sample.vision / 255
        hand_pos = hand_env.get_fingertip_positions()
        X[i], Y[i] = image, hand_pos

        if done:
            hand_env.reset()

    mean_pos = tf.reduce_mean(Y, axis=0)
    print(f"MSE from Mean: {tf.losses.mse(tf.reshape(Y, -1), tf.tile(mean_pos, [Y.shape[0]]))}")
    print(
        f"Euclidean Distance from Mean: {multi_point_euclidean_distance(tf.reshape(Y, -1), tf.tile(mean_pos, [Y.shape[0]]))}")

    # for i in range(20):
    #     index = random.randint(0, X.shape[0])
    #     plt.imshow(X[index, ...])
    #     plt.title(f"Image {index} ({Y[index, ...]})")
    #     plt.show()

    return tf.convert_to_tensor(X.astype("float32")), tf.convert_to_tensor(Y.astype("float32"))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    data = gen_cube_quats_prediction_data(1000)
