#!/usr/bin/env python
"""Pretrain the visual component."""
import argparse
import os
import sys

import numpy as np
import tensorflow_datasets as tfds
from tensorflow.python.data import AUTOTUNE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Tuple

import argcomplete
import tensorflow as tf

from angorapy.common.const import PRETRAINED_COMPONENTS_PATH
from angorapy.utilities.data_generation import gen_cube_quats_prediction_data, load_dataset


def pretrain_on_object_pose(pretrainable_component: tf.keras.Model,
                            epochs,
                            name="visual_op",
                            dataset: Tuple[np.ndarray, np.ndarray] = None,
                            load_from: str = None):
    """Pretrain a visual component on prediction of cube position."""
    if dataset is None:
        dataset = gen_cube_quats_prediction_data(1024 * 8, "storage/data/pretraining/pose_data.tfrecord")

    dataset = dataset.map(lambda x, y: (tf.image.per_image_standardization(x), y))
    dataset = dataset.shuffle(2000)

    testset = dataset.take(50000)
    trainset = dataset.skip(50000)
    valset, trainset = trainset.take(2000), trainset.skip(2000)

    trainset = trainset.prefetch(AUTOTUNE)
    trainset = trainset.batch(128)

    valset = valset.prefetch(AUTOTUNE)
    valset = valset.batch(128)

    testset = testset.prefetch(AUTOTUNE)
    testset = testset.batch(128)

    if load_from is None:
        model = pretrainable_component
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer, loss="mse", metrics=[])

        # train and save encoder
        model.fit(x=trainset,
                  epochs=epochs,
                  validation_data=valset,
                  callbacks=[
                      tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
                      tf.keras.callbacks.EarlyStopping(patience=6)],
                  shuffle=True
                  )
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}")
    else:
        print("Loading model...")
        model = tf.keras.models.load_model(load_from)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="mse", metrics=[])
        print("Model loaded successfully.")

    train_mean = np.mean(list(tfds.as_numpy(trainset.unbatch().take(10000).map(lambda x, y: y))), axis=0)
    test_numpy = np.stack(list(tfds.as_numpy(testset.unbatch().map(lambda x, y: y))))
    print(f"This model achieves {model.evaluate(testset)}")
    print(f"A mean model would achieve {np.mean((test_numpy - train_mean) ** 2)}")


if __name__ == "__main__":
    import keras_cortex as kc

    tf.get_logger().setLevel('INFO')
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("task", nargs="?", type=str, choices=["classify", "reconstruct", "hand", "object",
                                                              "c", "r", "h", "o", "hp", "op"], default="o")
    parser.add_argument("--name", type=str, default="visual_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--epochs", type=int, default=30, help=f"number of pretraining epochs")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # visual_component = _build_openai_encoder(shape=(VISION_WH, VISION_WH, 3), out_shape=7, name="visual_component")
    visual_component = kc.cornet.CORNetZ(output_dim=7, name=args.name)

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)

    args.name = args.name + "_" + args.task[0]

    pretrain_on_object_pose(
        visual_component, args.epochs,
        name=args.name,
        dataset=load_dataset("storage/data/pretraining/pose_data_200000.tfrecord"),
        load_from=args.load)
