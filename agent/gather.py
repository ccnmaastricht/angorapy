#!/usr/bin/env python
"""Gatherer classes."""
import itertools
import statistics
from abc import ABC, abstractmethod
from typing import List, Tuple

import gym
import numpy
import tensorflow as tf
from gym.spaces import Box

from agent.core import RLAgent, generalized_advantage_estimator, horizoned_generalized_advantage_estimator


class Gatherer(ABC):

    def __init__(self, environment: gym.Env, n_trajectories: int, normalize: bool = True):
        """Gatherer base class.

        :param environment:         a gym environment
        :param n_trajectories:      the number of desired trajectories
        """
        self.env = environment
        self.env_is_continuous = isinstance(self.env.action_space, Box)
        self.n_trajectories = n_trajectories

        # statistics
        self.total_frames = 0
        self.episode_reward_history = []
        self.episode_length_history = []
        self.last_episodes_completed = 0
        self.steps_during_last_gather = 0

        self.mean_episode_reward_per_gathering = []
        self.stdev_episode_reward_per_gathering = []

        self.normalize: bool = normalize

    @abstractmethod
    def gather(self, agent: RLAgent):
        pass

    def normalize_advantages(self, advantages: numpy.ndarray) -> numpy.ndarray:
        """Z-score standardization of advantages if activated in Gatherer."""
        if self.normalize:
            return (advantages - advantages.mean()) / advantages.std()
        return advantages


class ContinuousGatherer(Gatherer):

    def __init__(self, environment: gym.Env, horizon: int):
        """Continuous gatherer. Each trajectory goes until a fixed horizon.

        :param environment:
        :param horizon:                   the number of timesteps gathered
        """
        super().__init__(environment, 0)
        self.horizon = horizon

    def gather(self, agent: RLAgent) -> tf.data.Dataset:
        """Gather experience in an environment for n timesteps.

        :param agent:               the agent who is to be set into the environment

        :return:                    a 4-tuple where each element is a list of trajectories of s, r, a and p(a)
                                    where r[t] is the reward for taking a[t] in state s[t]
        """
        self.last_episodes_completed = 0
        self.steps_during_last_gather = 0

        state_trajectory = []
        reward_trajectory = []
        action_trajectory = []
        action_probability_trajectory = []
        t_is_terminal = []

        timestep = 1
        current_episode_return = 0
        state = tf.reshape(self.env.reset(), [1, -1])
        for t in range(self.horizon):
            self.total_frames += 1
            self.steps_during_last_gather += 1

            action, action_probability = agent.act(state)
            observation, reward, done, _ = self.env.step(
                numpy.atleast_1d(action.numpy()) if self.env_is_continuous else action.numpy()
            )

            # remember experience
            state_trajectory.append(tf.reshape(state, [-1]))
            reward_trajectory.append(reward)
            action_trajectory.append(action)
            action_probability_trajectory.append(action_probability)
            t_is_terminal.append(done == 1)
            current_episode_return += reward

            # next state
            if done:
                state = tf.reshape(self.env.reset(), [1, -1])
                self.episode_length_history.append(timestep)
                self.episode_reward_history.append(current_episode_return)
                self.last_episodes_completed += 1

                timestep = 1
                current_episode_return = 0
            else:
                state = tf.reshape(observation, [1, -1])
                timestep += 1

        self.mean_episode_reward_per_gathering.append(0 if self.last_episodes_completed < 1 else statistics.mean(
            self.episode_reward_history[-self.last_episodes_completed:]))
        self.stdev_episode_reward_per_gathering.append(0 if self.last_episodes_completed <= 1 else statistics.stdev(
            self.episode_reward_history[-self.last_episodes_completed:]))

        # value prediction needs to include last state that was not included too, in order to make GAE possible
        state_trajectory.append(state)
        value_predictions = [agent.critic_prediction(tf.reshape(s, [1, -1]))[0][0] for s in state_trajectory]
        advantages = horizoned_generalized_advantage_estimator(reward_trajectory, value_predictions, t_is_terminal,
                                                               gamma=agent.discount, gae_lambda=0.95)
        returns = numpy.add(advantages, value_predictions[:-1])

        advantages = self.normalize_advantages(advantages)

        return tf.data.Dataset.from_tensor_slices({
            "state": state_trajectory[:-1],
            "action": action_trajectory,
            "action_prob": action_probability_trajectory,
            "return": returns,
            "advantage": advantages
        })


@DeprecationWarning
class EpisodicGatherer(Gatherer):

    def gather(self, agent) -> Tuple[List, List, List, List, List]:
        """Gather experience in an environment for n trajectories.

        :param agent:               the agent who is to be set into the environment

        :return:                    a 4-tuple where each element is a list of trajectories of s, r, a and p(a)
        """
        self.last_episodes_completed = 0

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

            self.last_episodes_completed += 1

            state_trajectories.append(state_trajectory)
            reward_trajectories.append(reward_trajectory)
            action_probability_trajectories.append(action_probability_trajectory)
            action_trajectories.append(action_trajectory)

        state_value_predictions = [[agent.critic_prediction(tf.reshape(state, [1, -1]))[0][0]
                                    for state in trajectory] for trajectory in state_trajectories]
        advantages = [generalized_advantage_estimator(reward_trajectory, value_predictions,
                                                      gamma=agent.discount, gae_lambda=0.95)
                      for reward_trajectory, value_predictions in zip(reward_trajectories, state_value_predictions)]
        returns = numpy.add(advantages, state_value_predictions)

        # create the tensorflow dataset
        return tf.data.Dataset.from_tensor_slices({
            "state": list(itertools.chain(*state_trajectories)),
            "action": list(itertools.chain(*action_trajectories)),
            "action_prob": list(itertools.chain(*action_probability_trajectories)),
            "return": list(itertools.chain(*returns)),
            "advantage": list(itertools.chain(*advantages))
        })
