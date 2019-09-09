from __future__ import absolute_import, division, print_function, unicode_literals

import random
from abc import ABC, abstractmethod

import numpy
from tensorflow import keras

from agent.core import _RLAgent


class _QLearningAgent(_RLAgent, ABC):
    """Base Class for Q Learning Agents.
    Contains most functionality s.t. implementations only need to add model building.
    """

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        self.gamma = 0.95
        self.learning_rate = 0.001

        self.model = self._build_model()

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def act(self, state: numpy.ndarray, explorer=None):
        q_values = self.model.predict(state)

        if explorer is not None:
            return explorer.choose_action(q_values)
        else:
            return numpy.argmax(q_values)

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for experience, done in minibatch:
            if not done:
                # target for action taken by agent is predicted value of next state (when following policy) + reward
                focused_target = experience.reward + self.gamma * numpy.amax(
                    self.model.predict(experience.observation)[0])
            else:
                focused_target = experience.reward

            # target is current models prediction for all actions not taken by the agent -> no influence on loss
            target_vector = self.model.predict(experience.state)
            target_vector[0][experience.action] = focused_target

            # store in batch
            states.append(experience.state.reshape(-1))
            targets.append(target_vector.reshape(-1))

        # learn on batch
        self.model.fit(numpy.array(states), numpy.array(targets), epochs=1, verbose=0)


class LinearQLearningAgent(_QLearningAgent):

    def __init__(self, state_dimensionality, n_actions):
        super(LinearQLearningAgent, self).__init__(state_dimensionality, n_actions)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(self.n_actions, input_dim=self.state_dimensionality, activation="linear"))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model


class DeepQLearningAgent(_QLearningAgent):

    def __init__(self, state_dimensionality, n_actions):
        super(DeepQLearningAgent, self).__init__(state_dimensionality, n_actions)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_dimensionality, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.n_actions, activation="linear"))

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model


if __name__ == "__main__":
    agent = DeepQLearningAgent(4, 2)
