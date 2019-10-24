#!/usr/bin/env python
"""Components that can be loaded into another network."""
import os

import tensorflow as tf


class VisualComponent(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.convolutions = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 5, 1, input_shape=(200, 200, 3), activation="relu"),
            tf.keras.layers.Conv2D(32, 3, 1, activation="relu"),
            tf.keras.layers.MaxPooling2D(3, 3),
        ])

        self.res_blocks = tf.keras.Sequential([
            # following need to be changed to ResNet Blocks
            tf.keras.layers.Conv2D(64, 3, 3, activation="relu"),
            tf.keras.layers.Conv2D(64, 3, 3, activation="relu"),
        ])

        self.dense = tf.keras.Sequential([
            # TODO spatial softmax
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
        ])

    def call(self, input_tensor, training=False, **kwargs):
        x = self.convolutions(input_tensor)
        x = self.res_blocks(x)
        x = self.dense(x)

        return x


class VisualDecoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        spatial_reshape_size = 7 * 7 * 64
        self.expand = tf.keras.Sequential([
            # TODO spatial softmax
            tf.keras.layers.Dense(64, activation="relu", input_dim=32),
            tf.keras.layers.Dense(spatial_reshape_size, activation="relu"),
            tf.keras.layers.Reshape([7, 7, 64]),
        ])

        self.res_blocks = tf.keras.Sequential([
            # following need to be changed to ResNet Blocks
            tf.keras.layers.Conv2DTranspose(64, 3, 3, activation="relu"),
            tf.keras.layers.Conv2DTranspose(32, 3, 3, activation="relu", output_padding=1),
        ])

        self.convolutions = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, 3, 3, activation="relu", output_padding=2),
            tf.keras.layers.Conv2DTranspose(32, 3, 1, activation="relu"),
            tf.keras.layers.Conv2DTranspose(3, 5, 1, activation="relu"),
        ])

    def call(self, input_tensor, training=False, **kwargs):
        x = self.expand(input_tensor)

        x = self.res_blocks(x)
        x = self.convolutions(x)

        return x


class NonVisualComponent(tf.keras.Model):
    """For Proprioceptive and Somatosensory Input."""

    def __init__(self, input_dimensionality, output_dimensionality):
        super().__init__()

        hidden_dimensionality = output_dimensionality + ((input_dimensionality - output_dimensionality) // 2)
        self.fc_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(input_dimensionality, input_shape=(input_dimensionality,), activation="relu"),
            tf.keras.layers.Dense(hidden_dimensionality, activation="relu"),
            tf.keras.layers.Dense(output_dimensionality, activation="relu"),
        ])

    def call(self, input_tensor, training=False, **kwargs):
        return self.fc_layers(input_tensor)


class ResNetBlock(tf.keras.Model):
    """ResNet Block with variable number of convolutional sub-layers."""

    def __init__(self, filters, kernel_size, stride):
        super(ResNetBlock, self).__init__(name='')

        self.forward = tf.keras.Sequential()

        for n_filters in filters:
            self.forward.add(tf.keras.layers.Conv2D(n_filters, kernel_size, stride, padding="same"))
            self.forward.add(tf.keras.layers.BatchNormalization())
            self.forward.add(tf.keras.layers.ReLU())

    def call(self, input_tensor, training=False, **kwargs):
        x = self.forward(input_tensor)

        x += input_tensor
        return tf.nn.relu(x)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    net = VisualComponent()
    tensor = tf.random.normal([4, 200, 200, 3])
    latent = net(tensor)
    print(f"Latent Shape: {latent.shape}")

    decoder = VisualDecoder()
    reconstruction = decoder(latent)
    print(f"Reconstruction Shape: {reconstruction.shape}")
