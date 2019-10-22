#!/usr/bin/env python
"""TODO Module Docstring."""
import tensorflow as tf


class ShadowBrain(tf.keras.Model):

    def __init__(self, action_space):
        super().__init__()

        # PERCEPTIVE PROCESSORS

        # visual hyperparameters taken from OpenAI paper
        self.visual_component = tf.keras.Sequential(
            tf.keras.layers.Conv2D(32, 5, 1, input_shape=(200, 200, 3)),
            tf.keras.layers.Conv2D(32, 3, 1),
            tf.keras.layers.MaxPooling2D(3, 3),
            # following need to be changed to ResNet Blocks
            tf.keras.layers.Conv2D(16, 3, 3),
            tf.keras.layers.Conv2D(32, 3, 3),
            tf.keras.layers.Conv2D(64, 3, 3),
            tf.keras.layers.Conv2D(64, 3, 3),
            # TODO spatial softmax
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(32),
        )

        self.proprioceptive_component = tf.keras.Sequential(
            # input shape should depend on how many angles we use
            tf.keras.layers.Dense(24, input_shape=(24), activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
        )

        self.somatosensoric_component = tf.keras.Sequential(
            # input shape dependent on number of sensors
            tf.keras.layers.Dense(32, input_shape=(32), activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
        )

        # ABSTRACTION

        self.forward = tf.keras.Sequential(
            tf.keras.layers.Dense(32, input_shape=(32 + 8 + 6)),
            tf.keras.layers.Dense(32, input_shape=(32 + 8 + 6)),
        )

        self.recurrent_component = tf.keras.Sequential(
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(action_space, activation="softmax")
        )

    def call(self, inputs, training=None, mask=None):
        visual, proprio, somato, goal = inputs

        latent_visual = self.visual_component(visual)
        latent_proprio = self.proprioceptive_component(proprio)
        latent_somato = self.somatosensoric_component(somato)

        combined = tf.concat((latent_visual, latent_proprio, latent_somato), axis=1)
        abstracted = self.forward(combined)
        recurrent_input = tf.concat((abstracted, goal), axis=1)

        out = self.recurrent_component(recurrent_input)

        return out
