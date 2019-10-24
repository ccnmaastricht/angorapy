#!/usr/bin/env python
"""Pretrain the visual component."""
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from models.components import VisualComponent, VisualDecoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

test_train = tfds.load("cifar10", batch_size=32, shuffle_files=True)
images = test_train["test"].concatenate(test_train["train"])
images = images.map(lambda img: (tf.image.resize(img["image"], (200, 200)),) * 2)

autoencoder = tf.keras.Sequential([
    VisualComponent(),
    VisualDecoder()
])

optimizer = tf.keras.optimizers.Adam()
autoencoder.compile(optimizer, loss="mse")

autoencoder.fit(x=images)
