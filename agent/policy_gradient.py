import math
from abc import ABC
from typing import List, Any, Tuple

import numpy

from agent.core import _RLAgent

import tensorflow as tf
from tensorflow import keras

from datatypes import Experience


class _PolicyGradientAgent:

    def __init__(self):
        pass


# IMPLEMENTATIONS for specific ALGORITHM

class REINFORCEAgent(_PolicyGradientAgent):

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        # ENVIRONMENT
        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        # TRAINING PARAMETERS
        self.discount = 0.99
        self.learning_rate = 0.005

        # MODEL
        self.model = self._build_model()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_dimensionality, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.n_actions))
        model.add(keras.layers.Softmax())

        return model

    def loss(self, action_probability):
        return -tf.math.log(action_probability)

    def act(self, state):
        probabilities = self.model(state)
        action = numpy.random.choice(list(range(self.n_actions)), p=probabilities[0])

        return action, probabilities[0][action]


class ActorCriticREINFORCEAgent(REINFORCEAgent):

    def __init__(self, state_dimensionality, n_actions):
        super().__init__(state_dimensionality, n_actions)

        self.critic_lr = 0.001
        self.critic = self._build_value_model()
        self.critic_optimizer = keras.optimizers.Adam(lr=self.critic_lr)

    def _build_value_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_dimensionality, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(1))

        return model

    def critic_loss(self, value_prediction, future_reward):
        return future_reward - value_prediction

    def judge_state_value(self, state):
        return self.critic(state)

class TRPOAgent(_PolicyGradientAgent):

    def __init__(self):
        super().__init__()
        raise NotImplementedError


class PPOAgent(_PolicyGradientAgent):

    def __init__(self):
        super().__init__()
        raise NotImplementedError

