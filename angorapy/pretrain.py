#!/usr/bin/env python
"""Pretrain the visual component."""
import os
import sys

import mujoco

from angorapy.common.loss import PoseEstimationLoss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from angorapy import make_task
from angorapy.tasks.core import AnthropomorphicEnv

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

from mpi4py import MPI


mpi_rank = MPI.COMM_WORLD.Get_rank()
is_root = mpi_rank == 0

# deallocate the GPU from all but the root process
if not is_root:
    tf.config.set_visible_devices([], 'GPU')

# Prevent "Failed to find dnn implementation" error
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


hand_env = make_task("ManipulateBlockDiscreteAsynchronous-v0",
                     render_mode="rgb_array")
hand_env.reset()
# hand_env.render()

env_unwrapped: AnthropomorphicEnv = hand_env.unwrapped
model = env_unwrapped.model
data = env_unwrapped.data
renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)

cameras = [env_unwrapped._get_viewer("rgb_array").cam]
cameras[-1].distance = 0.3  # zoom in

for i in range(1, 3):
    cameras.append(mujoco.MjvCamera())
    cameras[-1].type = mujoco.mjtCamera.mjCAMERA_FREE
    cameras[-1].fixedcamid = -1
    for j in range(3):
        cameras[-1].lookat[j] = np.median(np.copy(data.geom_xpos[:, j]))
    cameras[-1].distance = np.copy(model.stat.extent)

    cameras[-1].distance = 0.3  # zoom in
    cameras[-1].azimuth = 0.0 + [20, -20][i - 1]  # wrist to the bottom
    cameras[-1].elevation = -90.0 + [20, -20][i - 1]  # top down view
    cameras[-1].lookat[0] = 0.35  # slightly move forward

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
    object = data.jnt("block/object:joint/")
    quat = data.jnt("block/object:joint/").qpos[3:]
    original_qpos = object.qpos.copy()

    if p < 0.2:  # Leave the object pose as is
        pass
    elif p < .6:  # Rotate the object by 90 degrees around its main axes
        # define the rotation
        axis = np.random.choice([0, 1, 2])
        sign = np.random.choice([0, 1])

        # rotate the body by 90 degrees around the chosen axis
        rotation_quat = rotation_quats[axis][sign]

        # Apply the rotation to the body quaternion
        new_quat = tfg.quaternion.multiply(quat, rotation_quat)
        object.qpos[3:] = new_quat
    else:  # "Jitter" the object by adding Gaussian noise to position and orientation
        # Add Gaussian noise to the object position
        noise_pos = np.random.normal(loc=0, scale=0.01, size=3)
        object.qpos[:3] += noise_pos

        # Add Gaussian noise to the object orientation
        noise_quat = np.random.normal(loc=0, scale=0.01, size=4)
        noise_quat /= np.linalg.norm(noise_quat)
        new_quat = tfg.quaternion.multiply(quat, noise_quat)
        object.qpos[3:] = new_quat

    mujoco.mj_forward(model, data)

    images = []
    for cam in cameras:
        renderer.update_scene(data, camera=cam)
        images.append(renderer.render().copy())

    qpos = object.qpos.copy()
    object.qpos = original_qpos

    images = tf.cast(tf.concat(images, axis=-1) / 255, dtype=tf.float32)
    qpos = tf.cast(qpos, dtype=tf.float32)

    # print(tf.reduce_max(images), tf.reduce_mean(images), tf.reduce_min(images))

    # plot slices of 3 channels in images next to each other
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, len(cameras))
    # for i in range(len(cameras)):
    #     ax = axs[i] if len(cameras) > 1 else axs
    #     ax.imshow(images[:, :, 3*i:3*(i+1)])
    # plt.show()
    # exit()

    return images, qpos


def pretrain_on_object_pose(pretrainable_component: tf.keras.Model,
                            epochs: int,
                            batch_size: int,
                            n_samples: int,
                            n_cameras=1,
                            load_data: bool = False,
                            name="visual_op",
                            load_from: str = None):
    """Pretrain a visual component on prediction of cube position."""
    data_name = f"storage/data/pretraining/pose_data_{n_samples}"
    data_path = f"{data_name}_{mpi_rank}.tfrecord"
    if not load_data:
        gen_cube_quats_prediction_data(
            n_samples,
            data_path,
        )

    if is_root:
        # get all filenames starting with data_name from the data directory
        filenames = tf.io.gfile.glob(f"{data_name}*.tfrecord")

        if len(filenames) == 0:
            raise FileNotFoundError(f"Could not find any files matching {data_name}*.tfrecord")

        dataset = load_unrendered_dataset(filenames)

        dataset = dataset.map(lambda x, y: tf.py_function(func=render, inp=[x], Tout=[tf.float32, tf.float32]))
        # dataset = dataset.map(lambda x, y: render(x))

        n_testset = 512
        n_valset = 256

        testset = dataset.take(n_testset)
        trainset = dataset.skip(n_testset)
        valset, trainset = trainset.take(n_valset), trainset.skip(n_valset)

        trainset = trainset.batch(batch_size, drop_remainder=True)
        valset = valset.batch(batch_size, drop_remainder=True)
        testset = testset.batch(batch_size, drop_remainder=True)

        if load_from is None:
            model = pretrainable_component

            build_sample = next(iter(trainset))
            model(build_sample[0])

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
            model.compile(optimizer, loss=PoseEstimationLoss(), metrics=["mse"])
            # model.compile(optimizer, loss="mse", metrics=[rotational_diff_metric, positional_diff_metric])

            def step_decay(epoch, lr):
                drop = 0.5
                epochs_drop = 1.0

                if epoch % epochs_drop == 0 and epoch > 1:
                    return lr * drop
                else:
                    return lr

            # train and save encoder
            model.fit(x=trainset,
                      epochs=epochs,
                      validation_data=valset,
                      callbacks=[
                          tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
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
        print(
            f"A mean model would achieve\n"
            f"\t{np.mean((test_numpy - train_mean) ** 2)} (mse) \n"
            f"\t{tf.reduce_mean(tf.norm(test_numpy[:, :3] - train_mean[:3], axis=-1, ord='euclidean'))} (pos: euclidean)."
        )


def pretrain_on_imagenet(pretrainable_component: tf.keras.Model,
                         epochs: int,
                         name="visual_op",
                         load_from: str = None,
                         batch_size: int = 128):
    """Pretrain a visual component on prediction of cube position."""
    trainset = tfds.load("imagenet_resized/64x64", split="train", shuffle_files=True, as_supervised=True)
    testset = tfds.load("imagenet_resized/64x64", split="validation", shuffle_files=True, as_supervised=True)

    valset, trainset = trainset.take(10000), trainset.skip(10000)

    trainset = trainset.batch(batch_size, drop_remainder=True)
    trainset = trainset.prefetch(AUTOTUNE)

    valset = valset.batch(batch_size, drop_remainder=True)
    valset = valset.prefetch(AUTOTUNE)

    testset = testset.batch(batch_size, drop_remainder=True)
    testset = testset.prefetch(AUTOTUNE)

    if load_from is None:
        model = pretrainable_component
        model(next(iter(trainset))[0])

        # chunk = list(tfds.as_numpy(dataset.take(8000).map(lambda x, y: y)))
        # chunk_mean = np.mean(chunk, axis=0)
        # output_layer = model.get_layer("output")
        # output_weights = output_layer.get_weights()
        # output_weights[1] = chunk_mean
        # output_layer.set_weights(output_weights)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy",
                                                                                  "sparse_top_k_categorical_accuracy"])

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
    tf.get_logger().setLevel('INFO')

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("task", nargs="?", type=str, choices=["classify", "reconstruct", "hand", "object",
                                                              "c", "r", "h", "o", "hp", "op"], default="o")
    parser.add_argument("--name", type=str, default="visual_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--epochs", type=int, default=25, help=f"number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=64, help=f"number of samples per minibatch")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # parameters
    mode = "pose"
    n_cameras = 3
    n_samples = 83392 // 2
    batch_size = args.batch_size

    visual_component = OpenAIEncoder(shape=(128, 128, 3), name=args.name, n_cameras=n_cameras, mode=mode)
    # visual_component = keras_cortex.cornet.cornet_z.PoseCORNetZ(name=args.name)

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)

    args.name = args.name + "_" + args.task[0]

    if mode == "classification":
        pretrain_on_imagenet(
            visual_component,
            epochs=args.epochs,
            name=args.name,
            load_from=args.load,
            batch_size=batch_size
        )
    elif mode == "pose":
        pretrain_on_object_pose(
            visual_component,
            epochs=args.epochs,
            batch_size=batch_size,
            n_samples=n_samples,
            n_cameras=n_cameras,
            load_data=True,
            name=args.name,
            load_from=args.load,
        )
    else:
        raise ValueError(f"Unknown mode {mode}.")