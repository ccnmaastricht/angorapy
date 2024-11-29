from angorapy.common.postprocessors import StateNormalizer
import keras

import tensorflow as tf


class StateNormalizationLayer(keras.layers.Layer):

    def __init__(self, mean, variance, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.variance = tf.constant(variance, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.clip_by_value((inputs - self.mean) / self.variance, -10., 10.)


def normalization_layer_from_postprocessor(modality: str, postprocessor: StateNormalizer):
    normalization_layer = StateNormalizationLayer(
        mean=postprocessor.mean[modality],
        variance=postprocessor.variance[modality],
        name=f"{modality}_normalization",
    )

    return normalization_layer
