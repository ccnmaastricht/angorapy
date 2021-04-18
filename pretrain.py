#!/usr/bin/env python
"""Pretrain the visual component."""
import argparse
import os
from typing import Union

import argcomplete
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import tensorflow_datasets as tfds

from models.convolutional import _build_openai_encoder, _build_visual_decoder
from utilities.const import PRETRAINED_COMPONENTS_PATH, VISION_WH
from utilities.data_generation import gen_cube_quats_prediction_data


def load_caltech():
    # load dataset
    test_train = tfds.load("caltech101", shuffle_files=True)
    return test_train["train"], test_train["test"], 102


def top_5_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


class TestCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        out = self.model.evaluate(self.test_data, )
        print(f"\n{out}\n")


def pretrain_on_reconstruction(pretrainable_component: Union[tf.keras.Model, str], epochs, name="visual_r"):
    """Pretrain a visual component on the reconstruction of images."""
    input_shape = pretrainable_component.input_shape
    spatial_dimensions = input_shape[1:3]

    X, _ = gen_cube_quats_prediction_data(1024 * 8)

    # model is constructed from visual component and a decoder
    decoder = _build_visual_decoder(pretrainable_component.output_shape[-1])
    model = tf.keras.Sequential((
        pretrainable_component,
        decoder
    ))

    if isinstance(pretrainable_component, tf.keras.Model):
        checkpoint_path = PRETRAINED_COMPONENTS_PATH + "/ckpts/weights.ckpt"

        # Create a callback that saves the model'serialization weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer, loss="mse")

        # train and save encoder
        model.fit(x=X, y=X, epochs=epochs, callbacks=[cp_callback])
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}.h5")
        model.save(PRETRAINED_COMPONENTS_PATH + f"/{name}_full.h5")
    elif isinstance(pretrainable_component, str):
        model.load_weights(pretrainable_component)
    else:
        raise ValueError("No clue what you think this is but it for sure ain't no model nor a path to model.")


def pretrain_on_classification(pretrainable_component: Union[tf.keras.Model, str], epochs, name="visual_c"):
    """Pretrain a visual component on the classification of images."""

    input_shape = pretrainable_component.input_shape
    spatial_dimensions = input_shape[1:3]

    train_images, test_images, n_classes = load_caltech()

    # resize and normalize images, extract one hot vectors from labels
    train_images = train_images.map(
        lambda img: (tf.image.resize(img["image"], spatial_dimensions) / 255, tf.one_hot(img["label"], depth=n_classes)))
    train_images = train_images.batch(128)

    test_images = test_images.map(
        lambda img: (tf.image.resize(img["image"], spatial_dimensions) / 255, tf.one_hot(img["label"], depth=n_classes)))
    test_images = test_images.batch(128)

    # model is constructed from visual component and classification layer
    model = tf.keras.Sequential((
        pretrainable_component,
        tf.keras.layers.Dense(n_classes),
        tf.keras.layers.Activation("softmax")
    ))

    if isinstance(pretrainable_component, tf.keras.Model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # train and save encoder
        model.fit(train_images, epochs=epochs, callbacks=[])
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}.h5")
        model.save(PRETRAINED_COMPONENTS_PATH + f"/{name}_full.h5")
    elif isinstance(pretrainable_component, str):
        model.load_weights(pretrainable_component)
    else:
        raise ValueError("No clue what you think this is but it for sure ain't no model nor a path to model.")

    # evaluate
    results = model.evaluate(test_images)
    print(results)


def pretrain_on_hands(pretrainable_component: Union[tf.keras.Model, str], epochs, name="visual_h"):
    """Pretrain a visual component on the classification of images."""

    # load datas
    X, Y = gen_cube_quats_prediction_data(1024 * 8)

    # model is constructed from visual component and regression output
    model = tf.keras.Sequential((
        pretrainable_component,
        tf.keras.layers.Dense(7, activation="linear"),
    ))

    if isinstance(pretrainable_component, tf.keras.Model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="mse", metrics=[])

        # train and save encoder
        model.fit(X, Y, epochs=epochs, callbacks=[], batch_size=128)
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}.h5")
    elif isinstance(pretrainable_component, str):
        model.load_weights(pretrainable_component)
    else:
        raise ValueError("No clue what you think this is but it for sure ain't no model nor a path to model.")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("task", nargs="?", type=str, choices=["classify", "reconstruct", "hands", "c", "r", "h"],
                        default="h")
    parser.add_argument("--name", type=str, default="pretrained_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--epochs", type=int, default=10, help=f"number of pretraining epochs")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    visual_component = _build_openai_encoder(shape=(VISION_WH, VISION_WH, 3), name="visual_component")

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)

    args.name = args.name + "_" + args.task[0]

    if args.task in ["classify", "c"]:
        pretrain_on_classification(visual_component, args.epochs, name=args.name)
    elif args.task in ["reconstruct", "r"]:
        pretrain_on_reconstruction(visual_component, args.epochs, name=args.name)
    elif args.task in ["hands", "h"]:
        pretrain_on_hands(visual_component, args.epochs, name=args.name)
    else:
        raise ValueError("I dont know that task type.")
