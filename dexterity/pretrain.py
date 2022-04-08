#!/usr/bin/env python
"""Pretrain the visual component."""
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Union

import argcomplete
import tensorflow as tf
import tensorflow_datasets as tfds

from dexterity.models import _build_openai_encoder, _build_visual_decoder, _build_openai_small_encoder
from common.const import PRETRAINED_COMPONENTS_PATH, VISION_WH
from utilities.data_generation import gen_cube_quats_prediction_data, gen_hand_pos_prediction_data
from common.loss import EuclideanDistanceLoss


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
        lambda img: (tf.image.resize(img["image"], spatial_dimensions) / 255,
                     tf.one_hot(img["label"], depth=n_classes)))
    train_images = train_images.batch(128)

    test_images = test_images.map(
        lambda img: (tf.image.resize(img["image"], spatial_dimensions) / 255,
                     tf.one_hot(img["label"], depth=n_classes)))
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


def pretrain_on_object_pose(pretrainable_component: Union[tf.keras.Model, str], epochs, name="visual_op"):
    """Pretrain a visual component on prediction of cube position."""

    # generate training data
    X, Y = gen_cube_quats_prediction_data(1024 * 8)
    X = tf.image.per_image_standardization(X)

    # model is constructed from visual component and regression output
    model = tf.keras.Sequential((
        pretrainable_component,
        tf.keras.layers.Dense(7, activation="linear"),
    ))

    if isinstance(pretrainable_component, tf.keras.Model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="mse", metrics=[EuclideanDistanceLoss])

        # train and save encoder
        model.fit(X, Y, epochs=epochs, callbacks=[], batch_size=128)
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}.h5")
    elif isinstance(pretrainable_component, str):
        model.load_weights(pretrainable_component)
    else:
        raise ValueError("No clue what you think this is but it for sure ain't no model nor a path to model.")


def pretrain_on_hand_pose(pretrainable_component: Union[tf.keras.Model, str], epochs, name="visual_hp"):
    """Pretrain a visual component on prediction of cube position."""

    # model is constructed from visual component and regression output
    model = pretrainable_component

    if isinstance(pretrainable_component, tf.keras.Model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        for epoch_i in range(epochs):
            print(f"Starting Epoch {epoch_i}")

            # generate training data
            X, Y = gen_hand_pos_prediction_data(1024 * 8)
            X = tf.image.per_image_standardization(X)

            dataset = tf.data.Dataset.from_tensor_slices((X, Y))
            dataset = dataset.shuffle(buffer_size=1024).batch(128)

            for step, (batch_x, batch_y) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    output = model(batch_x, training=True)
                    loss = tf.keras.losses.MSE(batch_y, output)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                if step % 4 == 0:
                    print(f"Training loss (for one batch) at step {step}: {float(tf.reduce_mean(loss))}. "
                          f"Seen so far: {((step + 1) * 128)} samples")

            del dataset

        # save encoder
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}.h5")
    elif isinstance(pretrainable_component, str):
        model.load_weights(pretrainable_component)
    else:
        raise ValueError("No clue what you think this is but it for sure ain't no model nor a path to a model.")


if __name__ == "__main__":
    tf.get_logger().setLevel('INFO')

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("task", nargs="?", type=str, choices=["classify", "reconstruct", "hand", "object",
                                                              "c", "r", "h", "o", "hp", "op"], default="h")
    parser.add_argument("--name", type=str, default="pretrained_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--epochs", type=int, default=60, help=f"number of pretraining epochs")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    visual_component = _build_openai_small_encoder(shape=(VISION_WH, VISION_WH, 3), out_shape=15, name="visual_component")
    visual_component(tf.random.normal((16, VISION_WH, VISION_WH, 3)))  # needed to initialize keras or whatever

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)

    args.name = args.name + "_" + args.task[0]

    if args.task in ["classify", "c"]:
        pretrain_on_classification(visual_component, args.epochs, name=args.name)
    elif args.task in ["reconstruct", "r"]:
        pretrain_on_reconstruction(visual_component, args.epochs, name=args.name)
    elif args.task in ["hand", "h"]:
        pretrain_on_hand_pose(visual_component, args.epochs, name=args.name)
    else:
        raise ValueError("I dont know that task type.")
