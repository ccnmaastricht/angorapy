#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time

import tensorflow as tf

from models.components import VisualComponent, NonVisualComponent


class ShadowBrain(tf.keras.Model):

    def __init__(self, action_space, goal_dim):
        super().__init__()

        # sensory input
        self.visual_component = VisualComponent()
        self.proprioceptive_component = NonVisualComponent(24, 8)
        self.somatosensory_component = NonVisualComponent(32, 8)

        # recurrent abstraction
        self.fc_layer = tf.keras.Sequential([tf.keras.layers.Dense(32, input_shape=(48 + goal_dim,)),
                                             tf.keras.layers.Dense(32)])
        self.recurrent_component = tf.keras.layers.LSTMCell(32)
        self.output_layer = tf.keras.layers.Dense(action_space, activation="softmax")

    def get_initial_state(self, batch_size, dtype=tf.float32):
        return self.recurrent_component.get_initial_state(batch_size=batch_size, dtype=dtype)

    def call(self, inputs, states, training=None, mask=None):
        visual, proprio, somato, goal = inputs

        latent_visual = self.visual_component(visual)
        latent_proprio = self.proprioceptive_component(proprio)
        latent_somato = self.somatosensory_component(somato)

        combined = tf.concat((latent_visual, latent_proprio, latent_somato, goal), axis=1)
        abstracted = self.fc_layer(combined)
        recurrent_input = tf.concat((abstracted, goal), axis=1)

        out, out_states = self.recurrent_component(recurrent_input, states)
        out = self.output_layer(out)

        return out, out_states


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    network = ShadowBrain(3, 4)
    hidden = network.get_initial_state(16)
    sequence = [(tf.random.normal([16, 200, 200, 3]),
                 tf.random.normal([16, 24]),
                 tf.random.normal([16, 32]),
                 tf.random.normal([16, 4])) for _ in range(10)]

    o = None
    start_time = time.time()
    for element in sequence:
        o, hidden = network(sequence[0], hidden)
    print(o)

    print(f"Execution Time: {time.time() - start_time}")
