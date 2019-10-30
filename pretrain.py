#!/usr/bin/env python
"""Pretrain the visual component."""
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from models.components import build_visual_component, build_visual_decoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOAD_FROM = None  # "saved_models/pretrained_components/visual_component/weights.ckpt"

# load dataset
test_train = tfds.load("cifar10", batch_size=128, shuffle_files=True)
images = test_train["test"].concatenate(test_train["train"])
images = images.map(lambda img: (tf.image.resize(img["image"], (200, 200)) / 255,) * 2)

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
