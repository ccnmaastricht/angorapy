#!/usr/bin/env python
"""Collection of fully connected policy networks."""

import tensorflow as tf


class PPOActorCriticNetwork(tf.keras.Model):
    """Fully-connected network with two heads: Actor and Critic.

    One head produces state-value predictions (critic), the other a probability distribution over actions (actor).
    This entails that the abstraction part of the network that theoretically is supposed to encode relevant information
    from the state has shared parameters, as the same information should be useful for both actor and critic. Only the
    analytical part is specific to the role.
    """

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        # shared base net
        self.fc_a = tf.keras.layers.Dense(16, input_dim=self.state_dimensionality, activation="relu")
        self.fc_b = tf.keras.layers.Dense(32, input_dim=16, activation="relu")
        self.fc_c = tf.keras.layers.Dense(12, input_dim=32, activation="relu")

        # role specific heads
        self.critic_head = tf.keras.layers.Dense(1, input_dim=12, activation="linear")
        self.actor_head = tf.keras.layers.Dense(self.n_actions, input_dim=12, activation="softmax")

    def call(self, input_tensor, training=False, **kwargs):
        x = self.fc_a(input_tensor)
        x = self.fc_b(x)
        x = self.fc_c(x)

        state_value = self.critic_head(x)
        action_probabilities = self.actor_head(x)

        return action_probabilities, state_value

