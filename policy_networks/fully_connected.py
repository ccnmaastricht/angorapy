#!/usr/bin/env python
"""Collection of fully connected policy networks."""
import gym
import tensorflow as tf

from utilities.util import env_extract_dims

tf.keras.backend.set_floatx("float64")


class PPOActorNetwork(tf.keras.Model):
    """Fully-connected network taking the role of an actor."""

    def __init__(self, env: gym.Env, continuous: bool):
        """

        :param env:
        :param continuous:      whether the action space predicted is continuous or not; if continuous, then the output
                                is not activated (linear) and the first n dimensions refer to the means and the last n
                                refer to the standard deviations where n is the number of actions.
        """
        super().__init__()

        self.continuous = continuous
        self.state_dimensionality, self.n_actions = env_extract_dims(env)

        self.fc_a = tf.keras.layers.Dense(64, input_dim=self.state_dimensionality, activation="tanh", dtype=tf.float64)
        self.fc_b = tf.keras.layers.Dense(64, input_dim=64, activation="tanh", dtype=tf.float64)
        self.fc_c = tf.keras.layers.Dense(32, input_dim=64, activation="tanh", dtype=tf.float64)

        self.fc_out = tf.keras.layers.Dense(self.n_actions,
                                            input_dim=32,
                                            activation="softmax" if not continuous else "linear",
                                            dtype=tf.float64)

        self.fc_stdev = tf.keras.layers.Dense(self.n_actions,
                                              input_dim=32,
                                              activation="relu",
                                              dtype=tf.float64)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.fc_a(input_tensor)
        x = self.fc_b(x)
        x = self.fc_c(x)

        out = self.fc_out(x)
        if self.continuous:
            stdev = self.fc_stdev(x) + 1e-10
            out = tf.concat((out, stdev), axis=1)

        return out


class PPOCriticNetwork(tf.keras.Model):
    """Fully-connected network taking the role of the critic."""

    def __init__(self, env):
        super().__init__()

        self.state_dimensionality, self.n_actions = env_extract_dims(env)

        # shared base net
        self.fc_a = tf.keras.layers.Dense(64, input_dim=self.state_dimensionality, activation="tanh", dtype=tf.float64)
        self.fc_b = tf.keras.layers.Dense(64, input_dim=64, activation="tanh", dtype=tf.float64)
        self.fc_c = tf.keras.layers.Dense(32, input_dim=64, activation="tanh", dtype=tf.float64)
        self.fc_out = tf.keras.layers.Dense(1, input_dim=32, activation="linear", dtype=tf.float64)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.fc_a(input_tensor)
        x = self.fc_b(x)
        x = self.fc_c(x)
        state_value = self.fc_out(x)

        return state_value


class PPOActorCriticNetwork(tf.keras.Model):
    """Fully-connected network with two heads: Actor and Critic.

    One head produces state-value predictions (critic), the other a probability distribution over actions (actor).
    This entails that the abstraction part of the network that theoretically is supposed to encode relevant information
    from the state has shared parameters, as the same information should be useful for both actor and critic. Only the
    analytical part is specific to the role.
    """

    def __init__(self, env: gym.Env, continuous: bool):
        super().__init__()

        self.state_dimensionality, self.n_actions = env_extract_dims(env)

        # shared base net
        self.fc_a = tf.keras.layers.Dense(16, input_dim=self.state_dimensionality, activation="tanh", dtype=tf.float64)
        self.fc_b = tf.keras.layers.Dense(32, input_dim=16, activation="tanh", dtype=tf.float64)
        self.fc_c = tf.keras.layers.Dense(12, input_dim=32, activation="tanh", dtype=tf.float64)

        # role specific heads
        self.fc_critic_a = tf.keras.layers.Dense(12, input_dim=12, activation="tanh", dtype=tf.float64)
        self.fc_critic_out = tf.keras.layers.Dense(1, input_dim=12, activation="linear", dtype=tf.float64)

        self.fc_actor_a = tf.keras.layers.Dense(12, input_dim=12, activation="tanh", dtype=tf.float64)
        self.fc_actor_out = tf.keras.layers.Dense(self.n_actions, input_dim=12,
                                                  activation="softmax" if not continuous else "linear",
                                                  dtype=tf.float64)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.fc_a(input_tensor)
        x = self.fc_b(x)
        x = self.fc_c(x)

        state_value = self.fc_critic_out(self.fc_critic_a(x))
        action_probabilities = self.fc_actor_out(self.fc_actor_a(x))

        return action_probabilities, state_value
