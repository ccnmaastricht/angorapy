#!/usr/bin/env python
"""TODO Module Docstring."""
import os

import tensorflow as tf

from models.components import VisualComponent


class ShadowBrain(tf.keras.Model):

    def __init__(self, action_space):
        super().__init__()

        # PERCEPTIVE PROCESSORS

        # visual hyperparameters taken from OpenAI paper
        self.visual_component = VisualComponent()

        self.proprioceptive_component = tf.keras.Sequential([
            # input shape should depend on how many angles we use
            tf.keras.layers.Dense(24, input_shape=(24,), activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
        ])

        self.somatosensoric_component = tf.keras.Sequential([
            # input shape dependent on number of sensors
            tf.keras.layers.Dense(32, input_shape=(32,), activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
        ])

        # ABSTRACTION

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(32 + 8 + 6,)),
            tf.keras.layers.Dense(32, input_shape=(32 + 8 + 6,)),
        ])

        self.recurrent_component = tf.keras.layers.LSTMCell(32)
        self.output_layer = tf.keras.layers.Dense(action_space, activation="softmax")

    def get_initial_state(self, batch_size, dtype=tf.float32):
        return self.recurrent_component.get_initial_state(batch_size=batch_size, dtype=dtype)

    def call(self, inputs, states, training=None, mask=None):
        visual, proprio, somato, goal = inputs

        latent_visual = self.visual_component(visual)
        latent_proprio = self.proprioceptive_component(proprio)
        latent_somato = self.somatosensoric_component(somato)

        combined = tf.concat((latent_visual, latent_proprio, latent_somato), axis=1)
        abstracted = self.forward(combined)
        recurrent_input = tf.concat((abstracted, goal), axis=1)

        out, out_states = self.recurrent_component(recurrent_input, states)
        out = self.output_layer(out)

        return out, out_states


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    network = ShadowBrain(3)
    hidden = network.get_initial_state(16)
    sequence = [(tf.random.normal([16, 200, 200, 3]),
                 tf.random.normal([16, 24]),
                 tf.random.normal([16, 32]),
                 tf.random.normal([16, 4])) for _ in range(10)]

    for element in sequence:
        o, hidden = network(sequence[0], hidden)
    print(o)
