#!/usr/bin/env python
"""Pretrain the visual component."""
import itertools
import os
import sys

import mujoco
import numpy as np
import tensorflow as tf
from angorapy.common.loss import PoseEstimationLoss
from angorapy.common.metrics.distance import distance_in_millimeters
from angorapy.common.metrics.distance import rotational_distance_in_degrees
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from angorapy import make_task
from angorapy.tasks.core import AnthropomorphicEnv
from angorapy.models.convolutional import OpenAIEncoder

import argparse

import tensorflow_datasets as tfds

import matplotlib

matplotlib.use('TkAgg')

import argcomplete

from angorapy.common.const import PRETRAINED_COMPONENTS_PATH, VISION_WH
from angorapy.utilities.data_generation import gen_cube_quats_prediction_data, load_unrendered_dataset

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

hand_env = make_task("TripleCamManipulateBlock-v0", render_mode="rgb_array")
hand_env.reset()
hand_env.render()

env_unwrapped: AnthropomorphicEnv = hand_env.unwrapped
model = env_unwrapped.model
data = env_unwrapped.data
renderer = hand_env.unwrapped.renderer
cameras = hand_env.unwrapped.cameras

rotation_quats = (
    (tfg.quaternion.from_axis_angle(np.array([[1., 0., 0.]]), np.array([[-1 * np.pi / 2]])),
     tfg.quaternion.from_axis_angle(np.array([[1., 0., 0.]]), np.array([[1 * np.pi / 2]]))),
    (tfg.quaternion.from_axis_angle(np.array([[0., 1., 0.]]), np.array([[-1 * np.pi / 2]])),
     tfg.quaternion.from_axis_angle(np.array([[0., 1., 0.]]), np.array([[1 * np.pi / 2]]))),
    (tfg.quaternion.from_axis_angle(np.array([[0., 0., 1.]]), np.array([[-1 * np.pi / 2]])),
     tfg.quaternion.from_axis_angle(np.array([[0., 0., 1.]]), np.array([[1 * np.pi / 2]])))
)


def render_from_sim_state(sim_state, augmentation_probabilities=(0.2, 0.4, 0.4)):
    """Render the scene from a given sim state."""
    env_unwrapped.data.qpos[:] = np.copy(sim_state[:model.nq])
    env_unwrapped.data.qvel[:] = np.copy(sim_state[model.nq:])

    if env_unwrapped.model.na == 0:
        env_unwrapped.data.act[:] = None

    mujoco.mj_forward(env_unwrapped.model, env_unwrapped.data)

    # Get the quaternion for the current orientation of the body
    block = data.jnt("block/object:joint/")
    quat = data.jnt("block/object:joint/").qpos[3:]

    original_qpos = block.qpos.copy()
    p = np.random.random()
    if p < augmentation_probabilities[0]:  # Leave the block pose as is
        pass
    elif p < sum(augmentation_probabilities[:2]):  # Rotate the block by 90 degrees around its main axes
        # define the rotation
        axis = np.random.choice([0, 1, 2])
        sign = np.random.choice([0, 1])

        # rotate the body by 90 degrees around the chosen axis
        rotation_quat = rotation_quats[axis][sign]

        # Apply the rotation to the body quaternion
        new_quat = tfg.quaternion.multiply(quat, rotation_quat)
        block.qpos[3:] = new_quat
    else:  # "Jitter" the block by adding Gaussian noise to position and orientation
        # Add Gaussian noise to the block position
        noise_pos = np.concatenate([np.random.normal(loc=0, scale=0.02, size=2), [0]])
        block.qpos[:3] += noise_pos

        # Add Gaussian noise to the block orientation
        noise_quat = np.random.normal(loc=0, scale=0.5, size=4)
        noise_quat /= np.linalg.norm(noise_quat)
        new_quat = tfg.quaternion.multiply(quat, noise_quat)
        block.qpos[3:] = new_quat

    mujoco.mj_forward(model, data)

    images = []
    for cam in cameras:
        renderer.update_scene(data, camera=cam)
        images.append(renderer.render())

    # plot the images
    # for i, image in enumerate(images):
    #     plt.subplot(1, len(images), i + 1)
    #     plt.imshow(image)
    #     plt.axis("off")
    #
    #     # set size
    #     fig = plt.gcf()
    #     fig.set_size_inches(18.5, 10.5)
    # plt.show()

    qpos = tf.cast(block.qpos.copy(), dtype=tf.float32)
    images = tf.cast(tf.concat(images, axis=-1), dtype=tf.float32)

    return images, qpos


def tf_render(sim_state, pose):
    output_shape = (VISION_WH, VISION_WH, 3 * len(cameras))

    images, qpos = render_from_sim_state(sim_state)
    images.set_shape(output_shape)

    return images, qpos


def prepare_dataset(name: str, load_data=True):
    """Prepare a dataset for pretraining."""
    # get all filenames starting with name from the data directory
    filenames = tf.io.gfile.glob(f"{name}*.tfrecord")

    if len(filenames) == 0:
        raise FileNotFoundError(f"Could not find any files matching {name}*.tfrecord")

    dataset = load_unrendered_dataset(filenames)

    return dataset


def pretrain_on_object_pose(pretrainable_component: tf.keras.Model,
                            n_batches: int,
                            batch_size: int,
                            n_samples: int,
                            n_cameras=3,
                            name="visual_op",
                            load_from: str = None):
    """Pretrain a visual component on prediction of cube position."""
    data_name = f"storage/data/pretraining/pose_data_{n_samples}"
    print(f"Preparing dataset {data_name}...")
    dataset = prepare_dataset(data_name, load_data=True)

    n_testset = 512 * 16
    n_valset = 512 * 2

    testset = dataset.take(n_testset)
    trainset = dataset.skip(n_testset)

    trainset = trainset.shuffle(512 * 64)
    trainset = trainset.repeat()

    trainset = trainset.batch(batch_size)
    testset = testset.batch(batch_size, drop_remainder=True)

    if is_root:
        train_mean = np.expand_dims(
            np.mean(list(tfds.as_numpy(trainset.unbatch().take(3000).map(lambda x, y: y))), axis=0), 0)
        test_y_numpy = np.stack(list(tfds.as_numpy(testset.unbatch().map(lambda x, y: y))))

        print(
            f"A mean model would achieve\n"
            f"\t{np.round(np.mean((test_y_numpy - train_mean) ** 2), 5)} (mse) \n"
            f"\t{np.round(PoseEstimationLoss()(test_y_numpy, train_mean), 5)} (pose loss) \n"
            f"\t{np.round(distance_in_millimeters(test_y_numpy, train_mean), 5)} mm (position). \n"
            f"\t{np.round(rotational_distance_in_degrees(test_y_numpy, train_mean), 5)} degrees (rotation).\n"
            f"based on the following mean: {train_mean}"
        )

        model = pretrainable_component

        build_sample = tf.expand_dims(render_from_sim_state(next(iter(testset))[0][0])[0], 0)
        model(build_sample)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        loss_fn = PoseEstimationLoss()
        metrics = [
            distance_in_millimeters,
            rotational_distance_in_degrees,
            keras.metrics.MeanSquaredError(),
        ]

    @tf.function
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss_value = loss_fn(y, predictions)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        metric_results = []
        for metric in metrics:
            metric_results.append(metric(tf.cast(y, dtype=tf.float32), predictions))

        return loss_value, *metric_results

    # train and save encoder
    report_every = 200

    batch_i = 0
    batch_losses = []
    windowed_mean_batch_losses = []
    for inputs, targets in tqdm(
            trainset,
            desc=f"Training on {n_batches} batches.",
            leave=False,
            total=n_batches,
            disable=not is_root
    ):

        # transform with render
        rendered_inputs, rendered_targets = [], []
        unstacked_data = tf.unstack(inputs, axis=0)
        worker_distributed_data = np.array_split(np.array(unstacked_data), MPI.COMM_WORLD.Get_size(), axis=0)[mpi_rank]

        for data in worker_distributed_data:
            rendered_data, rendered_pose = render_from_sim_state(data)
            rendered_inputs.append(rendered_data)
            rendered_targets.append(rendered_pose)

        # gather on root process
        if MPI.COMM_WORLD.Get_size() > 1:
            rendered_inputs = MPI.COMM_WORLD.gather(rendered_inputs, root=0)
            rendered_targets = MPI.COMM_WORLD.gather(rendered_targets, root=0)

        if is_root:
            # skip chaining if only one process is used
            if MPI.COMM_WORLD.Get_size() > 1:
                rendered_inputs = tf.stack(list(itertools.chain(*rendered_inputs)), axis=0)
                rendered_targets = tf.stack(list(itertools.chain(*rendered_targets)), axis=0)
            loss = np.array(train_on_batch(rendered_inputs, rendered_targets))
            batch_losses.append(loss)

            if batch_i % report_every == 0:
                windowed_mean_batch_losses.append(np.mean(batch_losses[-report_every:], axis=0))
                print(f"\rBatch {batch_i} - loss: {windowed_mean_batch_losses[-1]}")

                if batch_i >= report_every:
                    plt.plot(range(report_every, batch_i + 1, report_every),
                             np.array(windowed_mean_batch_losses)[1:, 0], label="loss")
                    plt.plot(range(report_every, batch_i + 1, report_every),
                             np.array(windowed_mean_batch_losses)[1:, 1], label="position")
                    plt.plot(range(report_every, batch_i + 1, report_every),
                             np.array(windowed_mean_batch_losses)[1:, 2], label="rotation")

                    plt.legend()
                    plt.savefig(f"storage/data/pretraining/{name}_loss.png")
                    plt.gcf().clear()

            if batch_i % 20000 == 0 and batch_i > 0:
                print(
                    f"Trained for {batch_i} batches. Saving model to {PRETRAINED_COMPONENTS_PATH}/{name}_{batch_i}...")
                pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}_{batch_i}")

                optimizer.learning_rate = optimizer.learning_rate / 2
                print(f"Learning rate reduced to {optimizer.learning_rate}")

        batch_i += 1

        if batch_i >= n_batches:
            break

    if is_root:
        print(f"Trained for {batch_i} batches. Saving model to {PRETRAINED_COMPONENTS_PATH}/{name}_{n_batches}...")
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}_{n_batches}")
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}")


def evaluate_on_object_pose(
        pretrainable_component: tf.keras.Model,
        dataset_name: str,
        n_datapoints: int,
        augmentation_probabilities=(1, 0, 0)
):
    """Pretrain a visual component on prediction of cube position."""
    if is_root:
        sim_states = []
        poses = []
        env = make_task("NoisyManipulateBlock-v0", render_mode="rgb_array")
        state, info = env.reset()
        for i in tqdm(range(n_datapoints)):
            state, reward, trunc, term, info = env.step(env.action_space.sample())
            pose = tf.cast(info["achieved_goal"], dtype=tf.float32)
            sim_state = env.get_state()
            sim_state = tf.cast(
                np.concatenate(
                    [np.copy(sim_state["qpos"]),
                     np.copy(sim_state["qvel"])]
                ), dtype=tf.float32
            )

            sim_states.append(sim_state)
            poses.append(pose)

            if trunc or term:
                state, info = env.reset()

        data = (sim_states, poses)
        dataset = tf.data.Dataset.from_tensor_slices(data)

        # data_name = f"storage/data/pretraining/{dataset_name}"
        # print(f"Preparing dataset {data_name}...")
        # dataset = prepare_dataset(data_name, load_data=True)

        dataset = dataset.shuffle(1024 * 32)
        testset = dataset.take(n_datapoints)
        testset = testset.batch(64, drop_remainder=True)

        model = pretrainable_component

        build_sample = tf.expand_dims(render_from_sim_state(next(iter(testset))[0][0])[0], 0)
        model(build_sample)

        metrics = [
            distance_in_millimeters,
            rotational_distance_in_degrees,
            keras.metrics.MeanSquaredError(),
        ]

    @tf.function
    def eval_on_batch(x, y):
        predictions = model(x, training=False)

        metric_results = []
        for metric in metrics:
            metric_results.append(metric(tf.cast(y, dtype=tf.float32), predictions))

        return metric_results

    batch_metric_results = []
    for inputs, targets in tqdm(
            testset,
            desc=f"Evaluating on {n_datapoints // 64} batches.",
            leave=False,
            total=n_datapoints // 64,
    ):

        # transform with render
        rendered_inputs, rendered_targets = [], []
        unstacked_data = tf.unstack(inputs, axis=0)

        for data in unstacked_data:
            rendered_data, rendered_pose = render_from_sim_state(
                data, augmentation_probabilities=augmentation_probabilities
            )

            rendered_inputs.append(rendered_data)
            rendered_targets.append(rendered_pose)

        metric_results = np.array(eval_on_batch(rendered_inputs, rendered_targets))
        batch_metric_results.append(metric_results)

    full_data_metrics = np.mean(batch_metric_results, axis=0)
    for metric, metric_result in zip(metrics, full_data_metrics):
        print(f"{metric}: {metric_result}")


if __name__ == "__main__":
    tf.get_logger().setLevel('INFO')

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("action", type=str, choices=["gen", "train", "eval", "all"], default="all")
    parser.add_argument("--name", type=str, default="visual_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--n-training-batches", type=int, default=400000, help=f"number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=64, help=f"number of samples per minibatch")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # parameters
    n_cameras = 3
    n_samples = 1000000

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)
    args.name = args.name

    # set up the visual component
    visual_component = OpenAIEncoder(shape=(VISION_WH, VISION_WH, 3), name=args.name, n_cameras=n_cameras)

    if args.action == "gen" or args.action == "all":
        # generate data
        name = data_name = f"storage/data/pretraining/pose_data_{n_samples}"
        path = f"{name}_{mpi_rank}.tfrecord"
        gen_cube_quats_prediction_data(
            n_samples // MPI.COMM_WORLD.Get_size(),
            path,
        )

    if args.action == "train" or args.action == "all":
        # pretrain the visual component
        pretrain_on_object_pose(
            visual_component,
            n_batches=args.n_training_batches,
            batch_size=args.batch_size,
            n_samples=n_samples,
            n_cameras=n_cameras,
            name=args.name,
            load_from=args.load,
        )

    if args.action == "eval" or args.action == "all":
        # evaluate the visual component
        visual_component = tf.keras.models.load_model(PRETRAINED_COMPONENTS_PATH
                                                      + f"/{args.name}"
                                                      + f"_{args.n_training_batches}")
        evaluate_on_object_pose(
            visual_component,
            dataset_name=f"pose_data_{n_samples}",
            n_datapoints=512 * 16,
            # augmentation_probabilities=(0.2, 0.4, 0.4),
        )
