#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import random
import statistics
from abc import ABC, abstractmethod

import numpy
from tensorflow import keras

from agent.core import RLAgent

from utilities.datatypes import Experience


class _QLearningAgent(RLAgent, ABC):
    """Base Class for Q Learning Agents.
    Contains most functionality s.t. implementations only need to add model building.
    """

    def __init__(self, state_dimensionality, n_actions, eps_init=0.1, eps_decay=0.999, eps_min=0.001):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        self.learning_rate = 0.001
        self.gamma = 0.95

        # epsilon greedy exploration parameters
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.epsilon = eps_init

        # function approximation model
        self.model = self._build_model()

        # replay buffer
        self.memory = collections.deque(maxlen=2000)

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def remember(self, experience, done):
        self.memory.append((experience, done))

    def act(self, state: numpy.ndarray):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            q_values = self.model.predict(state)
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

    def drill(self, env, n_iterations, batch_size=32, print_every=1000, avg_over=20):
        episode_rewards = [0]
        state = numpy.reshape(env.reset(), [1, -1])
        for iteration in range(n_iterations):
            # choose an action and make a step in the environment, based on the agents current knowledge
            action = self.act(numpy.array(state))
            observation, reward, done, info = env.step(action)
            episode_rewards[-1] += reward

            # reshape observation for tensorflow
            observation = numpy.reshape(observation, [1, -1])

            # remember experience made during the step in the agents memory
            self.remember(Experience(state, action, reward, observation), done)

            # log performance
            if iteration % print_every == 0:
                print(f"Iteration {iteration:10d}/{n_iterations} [{round(iteration/n_iterations * 100, 0)}%]"
                      f" | {len(episode_rewards) - 1:4d} episodes done"
                      f" | last {min(avg_over, len(episode_rewards)):2d}: "
                      f"avg {0 if len(episode_rewards) <= 1 else statistics.mean(episode_rewards[-avg_over:-1]):5.2f}; "
                      f"max {0 if len(episode_rewards) <= 1 else max(episode_rewards[-avg_over:-1]):5.2f}; "
                      f"min {0 if len(episode_rewards) <= 1 else min(episode_rewards[-avg_over:-1]):5.2f}"
                      f" | epsilon: {self.epsilon}")

            if len(self.memory) > batch_size:
                # learn from memory
                self.learn(batch_size)
                # update exploration
                self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

            # if episode ended reset env and rewards, otherwise go to next state
            if done:
                state = numpy.reshape(env.reset(), [1, -1])
                episode_rewards.append(0)
            else:
                state = observation


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
    self = DeepQLearningAgent(4, 2)
