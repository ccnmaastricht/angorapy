#!/usr/bin/env python
"""Pretrain the visual component."""
import argparse
import os
from typing import Union

import argcomplete
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from models.convolutional import _build_visual_encoder, _build_visual_decoder
from utilities.const import PRETRAINED_COMPONENTS_PATH


def pretrain_on_reconstruction(pretrainable_component: Union[tf.keras.Model, str], epochs, name="pretrained_component"):
    """Pretrain a visual component on the reconstruction of images."""
    input_shape = pretrainable_component.input_shape
    spatial_dimensions = input_shape[1:3]

    # load dataset
    test_train = tfds.load("cifar10", shuffle_files=True)
    test_images = test_train["test"]
    train_images = test_train["train"]

    images = test_images.concatenate(train_images).map(
        lambda img: (tf.image.resize(img["image"], spatial_dimensions) / 255,) * 2).batch(128)

    # model is constructed from visual component and a decoder
    decoder = _build_visual_decoder(pretrainable_component.output_shape[-1])
    model = tf.keras.Sequential((
        pretrainable_component,
        decoder
    ))

    if isinstance(pretrainable_component, tf.keras.Model):
        checkpoint_path = PRETRAINED_COMPONENTS_PATH + "/ckpts/weights.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer, loss="mse")

        # train and save encoder
        model.fit(x=images, epochs=epochs, callbacks=[cp_callback])
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}.h5")
    elif isinstance(pretrainable_component, str):
        model.load_weights(pretrainable_component)
    else:
        raise ValueError("No clue what you think this is but it for sure ain't no model nor a path to model.")

    # inspect
    inspection_images = images.unbatch().take(10)
    for img, _ in inspection_images:
        prediction = model.predict(tf.expand_dims(img, axis=0))
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(img.numpy())
        axes[1].imshow(tf.squeeze(prediction).numpy())

        plt.show()


def pretrain_on_classification(pretrainable_component: Union[tf.keras.Model, str], epochs, name="pretrained_component"):
    """Pretrain a visual component on the classification of images."""

    input_shape = pretrainable_component.input_shape
    spatial_dimensions = input_shape[1:3]

    # load dataset
    test_train = tfds.load("cifar10", shuffle_files=True)
    test_images = test_train["test"]
    train_images = test_train["train"]

    # resize and normalize images, extract one hot vectors from labels
    train_images = train_images.map(
        lambda img: (tf.image.resize(img["image"], spatial_dimensions) / 255, tf.one_hot(img["label"], depth=10)))
    train_images = train_images.batch(128)

    test_images = test_images.map(
        lambda img: (tf.image.resize(img["image"], spatial_dimensions) / 255, tf.one_hot(img["label"], depth=10)))
    test_images = test_images.batch(128)

    # model is constructed from visual component and classification layer
    model = tf.keras.Sequential((
        pretrainable_component,
        tf.keras.layers.Dense(10, activation="softmax")
    ))

    if isinstance(pretrainable_component, tf.keras.Model):
        checkpoint_path = PRETRAINED_COMPONENTS_PATH + "/ckpts/weights.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # train and save encoder
        model.fit(train_images, epochs=epochs, callbacks=[cp_callback])
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}.h5")
    elif isinstance(pretrainable_component, str):
        model.load_weights(pretrainable_component)
    else:
        raise ValueError("No clue what you think this is but it for sure ain't no model nor a path to model.")

    # evaluate
    results = model.evaluate(test_images)
    print(results)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("task", nargs="?", type=str, choices=["classify", "reconstruct", "c", "r"], default="c")
    parser.add_argument("--name", type=str, default="pretrained_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--epochs", type=int, default=10, help=f"number of pretraining epochs")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    visual_component = _build_visual_encoder(shape=(227, 227, 3), name="visual_component")

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)

    if args.task in ["classify", "c"]:
        pretrain_on_classification(visual_component, args.epochs, name=args.name)
    elif args.task in ["reconstruct", "r"]:
        pretrain_on_reconstruction(visual_component, args.epochs, name=args.name)
    else:
        raise ValueError("I dont know that task type.")
