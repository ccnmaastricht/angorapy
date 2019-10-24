#!/usr/bin/env python
"""Custom layer components to be used in model implementations."""
import tensorflow as tf


class SpatialSoftmax(tf.keras.Model):
    """Spatial Softmax Layer."""

    def __init__(self):
        super().__init__()

    def call(self, inputs, training=None, mask=None):
        pass

