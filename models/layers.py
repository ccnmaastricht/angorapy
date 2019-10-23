#!/usr/bin/env python
"""Custom layer components to be used in model implementations."""
import tensorflow as tf


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


class SpatialSoftmax(tf.keras.Model):
    """Spatial Softmax Layer."""

    def __init__(self):
        super().__init__()

    def call(self, inputs, training=None, mask=None):
        pass

