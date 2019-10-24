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


checkpoint_dir = "saved_models/pretrained_components/visual_component/"
os.makedirs(checkpoint_dir)
checkpoint_path = checkpoint_dir + "/weights.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


optimizer = tf.keras.optimizers.Adam()
autoencoder.compile(optimizer, loss="mse")

autoencoder.fit(x=images, epochs=20, callbacks=[cp_callback])
