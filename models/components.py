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
            tf.keras.layers.Conv2D(16, 3, 3, activation="relu"),
            tf.keras.layers.Conv2D(32, 3, 3, activation="relu"),
        ])

        self.dense = tf.keras.Sequential([
            # TODO spatial softmax
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
        ])

    def __call__(self, input_tensor, training=False, **kwargs):
        x = self.convolutions(input_tensor)
        x = self.res_blocks(x)
        x = self.dense(x)

        return x


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

    print(net(tensor))
