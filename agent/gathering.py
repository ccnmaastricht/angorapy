#!/usr/bin/env python
"""Gatherer classes."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import gym

import tensorflow as tf

from agent.core import _RLAgent


class _Gatherer(ABC):

    def __init__(self, environment: gym.Env, n_trajectories: int):
        """Gatherer base class.

        :param environment:
        :param n_trajectories:      the number of desired trajectories
        """
        self.env = environment
        self.n_trajectories = n_trajectories

    @abstractmethod
    def gather(self, agent: _RLAgent):
        pass


class EpisodicGatherer(_Gatherer):

    def gather(self, agent) -> Tuple[List, List, List, List, List]:
        """Gather experience in an environment for n trajectories.

        :param agent:               the agent who is to be set into the environment

        :return:                    a 4-tuple where each element is a list of trajectories of s, r, a and p(a)
        """
        state_trajectories = []
        reward_trajectories = []
        action_trajectories = []
        action_probability_trajectories = []

        for episode in range(self.n_trajectories):
            state_trajectory = []
            reward_trajectory = []
            action_trajectory = []
            action_probability_trajectory = []

            done = False
            state = tf.reshape(self.env.reset(), [1, -1])
            while not done:
                action, action_probability = agent.act(state)
                observation, reward, done, _ = self.env.step(action.numpy())

                # remember experience
                state_trajectory.append(tf.reshape(state, [-1]))  # does not incorporate the state inducing DONE
                reward_trajectory.append(reward)
                action_trajectory.append(action)
                action_probability_trajectory.append(action_probability)

                # next state
                state = tf.reshape(observation, [1, -1])

            state_trajectories.append(state_trajectory)
            reward_trajectories.append(reward_trajectory)
            action_probability_trajectories.append(action_probability_trajectory)
            action_trajectories.append(action_trajectory)

        return state_trajectories, reward_trajectories, action_trajectories, action_probability_trajectories, []


class ContinuousGatherer(_Gatherer):

    def __init__(self, environment: gym.Env, horizon: int):
        """Continuous gatherer. Each trajectory goes until a fixed horizon.

        :param environment:
        :param horizon:                   the number of timesteps gathered
        """
        super().__init__(environment, 0)
        self.horizon = horizon

    def gather(self, agent) -> Tuple[List, List, List, List, List]:
        """Gather experience in an environment for n timesteps.

        :param agent:               the agent who is to be set into the environment

        :return:                    a 4-tuple where each element is a list of trajectories of s, r, a and p(a)
        """
        state_trajectory = []
        reward_trajectory = []
        action_trajectory = []
        action_probability_trajectory = []
        terminal_state_indices = []

        state = tf.reshape(self.env.reset(), [1, -1])
        for t in range(self.horizon):
            action, action_probability = agent.act(state)
            observation, reward, done, _ = self.env.step(action.numpy())

            # remember experience
            state_trajectory.append(tf.reshape(state, [-1]))
            reward_trajectory.append(reward)
            action_trajectory.append(action)
            action_probability_trajectory.append(action_probability)

            # next state
            if done:
                terminal_state_indices.append(t)
                state = tf.reshape(self.env.reset(), [1, -1])
            else:
                state = tf.reshape(observation, [1, -1])

        return [state_trajectory], [reward_trajectory], [action_trajectory], [action_probability_trajectory], \
               [terminal_state_indices]
