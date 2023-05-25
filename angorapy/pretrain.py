#!/usr/bin/env python
"""Pretrain the visual component."""
import os
import sys

import mujoco
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from angorapy import make_env
from angorapy.environments.anthrobotics import AnthropomorphicEnv

from angorapy.models.convolutional import OpenAIEncoder

import argparse
import math

import numpy as np
import tensorflow_datasets as tfds
from tensorflow.python.data import AUTOTUNE

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Tuple

import argcomplete
import tensorflow as tf

from angorapy.common.const import PRETRAINED_COMPONENTS_PATH, VISION_WH
from angorapy.utilities.data_generation import gen_cube_quats_prediction_data, load_dataset, load_unrendered_dataset

import tensorflow_graphics.geometry.transformation as tfg

# tf.get_logger().setLevel('INFO')
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


@tf.function
def rotational_diff_metric(y_true, y_pred):
    rot_true = y_true[..., 3:]
    rot_pred = y_pred[..., 3:]

    return tfg.quaternion.relative_angle(rot_true, rot_pred) * tf.constant(180. / math.pi)


@tf.function
def positional_diff_metric(y_true, y_pred):
    """Gives positional difference in millimeters."""
    pos_true = y_true[..., :3]
    pos_pred = y_pred[..., :3]

    return tf.linalg.norm(pos_true - pos_pred, axis=-1) * 1000


hand_env = make_env("HumanoidVisualManipulateBlockDiscreteAsynchronous-v0",
                    render_mode="rgb_array")

env_unwrapped: AnthropomorphicEnv = hand_env.unwrapped
model = env_unwrapped.model
data = env_unwrapped.data
renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)
cameras = [env_unwrapped._get_viewer("rgb_array").cam]
for i in range(1, 3):
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

rotation_quats = (
    (tfg.quaternion.from_axis_angle(np.array([[1., 0., 0.]]), np.array([[-1 * np.pi / 2]])),
     tfg.quaternion.from_axis_angle(np.array([[1., 0., 0.]]), np.array([[1 * np.pi / 2]]))),
    (tfg.quaternion.from_axis_angle(np.array([[0., 1., 0.]]), np.array([[-1 * np.pi / 2]])),
     tfg.quaternion.from_axis_angle(np.array([[0., 1., 0.]]), np.array([[1 * np.pi / 2]]))),
    (tfg.quaternion.from_axis_angle(np.array([[0., 0., 1.]]), np.array([[-1 * np.pi / 2]])),
     tfg.quaternion.from_axis_angle(np.array([[0., 0., 1.]]), np.array([[1 * np.pi / 2]])))
)


def render(sim_state):
    hand_env.set_state(sim_state[:model.nq], sim_state[model.nq:])

    # Choose a random number between 0 and 1
    p = np.random.random()

    # Get the quaternion for the current orientation of the body
    object = data.jnt("object:joint")
    quat = data.jnt("object:joint").qpos[3:]
    original_qpos = object.qpos.copy()

    if p < 0.0:  # Leave the object pose as is
        pass
    elif p < 1.0:  # Rotate the object by 90 degrees around its main axes
        # define the rotation
        axis = np.random.choice([0, 1, 2])
        sign = np.random.choice([0, 1])

        # rotate the body by 90 degrees around the chosen axis
        rotation_quat = rotation_quats[axis][sign]

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

    images = tf.cast(tf.concat(images, axis=-1) / 255, dtype=tf.float32)
    qpos = tf.cast(qpos, dtype=tf.float32)

    return images, qpos


def pretrain_on_object_pose(pretrainable_component: tf.keras.Model,
                            epochs: int,
                            n_samples: int,
                            n_cameras=1,
                            load_data: bool = False,
                            name="visual_op",
                            load_from: str = None):
    """Pretrain a visual component on prediction of cube position."""
    data_path = f"storage/data/pretraining/pose_data_{n_samples}_{n_cameras}c.tfrecord"
    if not load_data:
        dataset = gen_cube_quats_prediction_data(
            n_samples,
            data_path,
        )
    else:
        dataset = load_unrendered_dataset(data_path)

    dataset = dataset.map(lambda x, y: tf.py_function(func=render, inp=[x], Tout=[tf.float32, tf.float32]))
    # dataset = dataset.map(lambda x, y: render(x))

    n_testset = 1000
    n_valset = 500

    testset = dataset.take(n_testset)
    trainset = dataset.skip(n_testset)
    valset, trainset = trainset.take(n_valset), trainset.skip(n_valset)

    trainset = trainset.batch(1, drop_remainder=True, num_parallel_calls=1)
    trainset = trainset.prefetch(AUTOTUNE)

    valset = valset.batch(1, drop_remainder=True, num_parallel_calls=1)
    valset = valset.prefetch(AUTOTUNE)

    testset = testset.batch(1, drop_remainder=True, num_parallel_calls=1)
    testset = testset.prefetch(AUTOTUNE)

    if load_from is None:
        model = pretrainable_component

        build_sample = tf.expand_dims(next(iter(dataset))[0], 0)
        model(build_sample)

        # chunk = list(tfds.as_numpy(dataset.take(8000).map(lambda x, y: y)))
        # chunk_mean = np.mean(chunk, axis=0)
        # output_layer = model.get_layer("output")
        # output_weights = output_layer.get_weights()
        # output_weights[1] = chunk_mean
        # output_layer.set_weights(output_weights)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer, loss="mse", metrics=[])
        # model.compile(optimizer, loss="mse", metrics=[rotational_diff_metric, positional_diff_metric])

        # train and save encoder
        model.fit(x=trainset,
                  epochs=epochs,
                  validation_data=valset,
                  # callbacks=[
                  #     tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
                  # ],
                  shuffle=True)
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}")
    else:
        print("Loading model...")
        model = tf.keras.models.load_model(load_from)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="mse", metrics=[rotational_diff_metric, positional_diff_metric])
        print("Model loaded successfully.")

    train_mean = np.mean(list(tfds.as_numpy(trainset.unbatch().take(5000).map(lambda x, y: y))), axis=0)
    test_numpy = np.stack(list(tfds.as_numpy(testset.unbatch().map(lambda x, y: y))))
    print(f"This model achieves {model.evaluate(testset)}")
    print(f"A mean model would achieve {np.mean((test_numpy - train_mean) ** 2)}")


def pretrain_on_rendered_object_pose(pretrainable_component: tf.keras.Model,
                                     epochs: int,
                                     n_samples: int,
                                     n_cameras=1,
                                     load_data: bool = False,
                                     name="visual_op",
                                     dataset: Tuple[np.ndarray, np.ndarray] = None,
                                     load_from: str = None):
    """Pretrain a visual component on prediction of cube position."""
    data_path = f"storage/data/pretraining/pose_data_{n_samples}_{n_cameras}c.tfrecord"
    if not load_data:
        dataset = gen_cube_quats_prediction_data(
            n_samples,
            data_path,
            n_cameras=n_cameras
        )
    else:
        dataset = load_dataset(data_path)

    # dataset = dataset.repeat(100000).shuffle(10000)
    # dataset = dataset.map(lambda x, y: (x, y))

    n_testset = 10000
    n_valset = 5000

    testset = dataset.take(n_testset)
    trainset = dataset.skip(n_testset)
    valset, trainset = trainset.take(n_valset), trainset.skip(n_valset)

    trainset = trainset.batch(128, drop_remainder=True)
    trainset = trainset.prefetch(AUTOTUNE)

    valset = valset.batch(128, drop_remainder=True)
    valset = valset.prefetch(AUTOTUNE)

    testset = testset.batch(128, drop_remainder=True)
    testset = testset.prefetch(AUTOTUNE)

    print(next(iter(dataset))[0])

    if load_from is None:
        model = pretrainable_component
        model(tf.expand_dims(next(iter(dataset))[0], 0))

        # chunk = list(tfds.as_numpy(dataset.take(8000).map(lambda x, y: y)))
        # chunk_mean = np.mean(chunk, axis=0)
        # output_layer = model.get_layer("output")
        # output_weights = output_layer.get_weights()
        # output_weights[1] = chunk_mean
        # output_layer.set_weights(output_weights)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer, loss="mse", metrics=[rotational_diff_metric, positional_diff_metric])

        # train and save encoder
        model.fit(x=trainset,
                  epochs=epochs,
                  validation_data=valset,
                  callbacks=[
                      tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
                  ],
                  shuffle=True)
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}")
    else:
        print("Loading model...")
        model = tf.keras.models.load_model(load_from)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="mse", metrics=[rotational_diff_metric, positional_diff_metric])
        print("Model loaded successfully.")

    train_mean = np.mean(list(tfds.as_numpy(trainset.unbatch().take(5000).map(lambda x, y: y))), axis=0)
    test_numpy = np.stack(list(tfds.as_numpy(testset.unbatch().map(lambda x, y: y))))
    print(f"This model achieves {model.evaluate(testset)}")
    print(f"A mean model would achieve {np.mean((test_numpy - train_mean) ** 2)}")


if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("task", nargs="?", type=str, choices=["classify", "reconstruct", "hand", "object",
                                                              "c", "r", "h", "o", "hp", "op"], default="o")
    parser.add_argument("--name", type=str, default="visual_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--epochs", type=int, default=2, help=f"number of pretraining epochs")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # parameters
    n_cameras = 3
    n_samples = 10000

    visual_component = OpenAIEncoder(shape=(128, 128, 3), name=args.name, n_cameras=n_cameras)
    # visual_component = keras_cortex.cornet.cornet_z.PoseCORNetZ(7, name=args.name)

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)

    args.name = args.name + "_" + args.task[0]

    pretrain_on_object_pose(
        visual_component,
        epochs=args.epochs,
        n_samples=n_samples,
        n_cameras=n_cameras,
        load_data=True,
        name=args.name,
        load_from=args.load,
    )
