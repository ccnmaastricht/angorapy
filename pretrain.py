#!/usr/bin/env python
"""Pretrain the visual component."""
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.utils import to_categorical

from models.components import build_visual_component, build_visual_decoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOAD_FROM = None  # "saved_models/pretrained_components/visual_component/weights.ckpt"
TASK = "classification"

# load dataset
test_train = tfds.load("cifar10", shuffle_files=True)
test_images = test_train["test"]
train_images = test_train["train"]

if TASK == "reconstruction":
    images = test_images.concatenate(train_images).map(
        lambda img: (tf.image.resize(img["image"], (200, 200)) / 255,) * 2)

    images = images.batch(128)

    # model is constructed from visual component and a decoder
    inputs = tf.keras.Input(shape=(200, 200, 3))
    encoder = build_visual_component()
    decoder = build_visual_decoder()
    encoded_image = encoder(inputs)
    decoded_image = decoder(encoded_image)
    model = tf.keras.Model(inputs=inputs, outputs=decoded_image)
    model.summary()

    if LOAD_FROM is None:
        # model with decoder
        checkpoint_dir = "saved_models/pretrained_components/visual_component/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = checkpoint_dir + "/weights.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer, loss="mse")

        # train and save encoder
        model.fit(x=images, epochs=30, callbacks=[cp_callback])
        encoder.save(checkpoint_dir + "/pretrained_encoder.h5")
    else:
        model.load_weights(LOAD_FROM)

    # inspect
    inspection_images = images.unbatch().take(10)
    for img, _ in inspection_images:
        prediction = model.predict(tf.expand_dims(img, axis=0))
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(img.numpy())
        axes[1].imshow(tf.squeeze(prediction).numpy())

        plt.show()
elif TASK == "classification":
    train_images = train_images.map(
        lambda img: (tf.image.resize(img["image"], (200, 200)) / 255, tf.one_hot(img["label"], depth=10)))
    train_images = train_images.batch(128)

    test_images = test_images.map(
        lambda img: (tf.image.resize(img["image"], (200, 200)) / 255, tf.one_hot(img["label"], depth=10)))
    test_images = test_images.batch(128)

    # model is constructed from visual component and classification layer
    inputs = tf.keras.Input(shape=(200, 200, 3))
    encoder = build_visual_component()
    encoded_img = encoder(inputs)
    classifier = tf.keras.layers.Dense(10, activation="softmax")(encoded_img)
    model = tf.keras.Model(inputs=inputs, outputs=classifier)

    if LOAD_FROM is None:
        # model with decoder
        checkpoint_dir = "saved_models/pretrained_components/visual_component/classification/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = checkpoint_dir + "/weights.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # train and save encoder
        model.fit(train_images, epochs=5, callbacks=[cp_callback])
        encoder.save(checkpoint_dir + "/pretrained_encoder.h5")
    else:
        model.load_weights(LOAD_FROM)

    # evaluate
    results = model.evaluate(test_images)
    print(results)
