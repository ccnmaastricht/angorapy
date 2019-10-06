#!/usr/bin/env python
"""Gatherer classes."""
import copy
import statistics

import gym
import numpy
import tensorflow as tf
from gym.spaces import Box

from agent.core import RLAgent, estimate_advantage


class Gatherer:

    def __init__(self, environment: gym.Env, horizon: int, workers: int = 1, normalize: bool = True):
        """Gatherer base class.

        :param environment:         a gym environment
        :param horizon:             the number of timesteps to be generated per worker
        :param workers:
        """
        assert workers >= 1

        self.env = environment
        self.env_is_continuous = isinstance(self.env.action_space, Box)

        self.horizon = horizon
        self.workers = workers
        self.normalize: bool = normalize

        # statistics
        self.total_frames = 0
        self.episode_reward_history = []
        self.episode_length_history = []
        self.last_episodes_completed = 0
        self.steps_during_last_gather = 0

        self.mean_episode_reward_per_gathering = []
        self.stdev_episode_reward_per_gathering = []

    def normalize_advantages(self, advantages: numpy.ndarray) -> numpy.ndarray:
        """Z-score standardization of advantages if activated."""
        if self.normalize:
            return (advantages - advantages.mean()) / advantages.std()
        return advantages

    def gather(self, agent: RLAgent) -> tf.data.Dataset:
        """Gather experience in an environment for n timesteps.

        :param agent:               the agent who is to be set into the environment

        :return:                    a 4-tuple where each element is a list of trajectories of s, r, a and p(a)
                                    where r[t] is the reward for taking a[t] in state s[t]
        """
        shared_s_trajectory, shared_a_trajectory, shared_aprob_trajectory, shared_r, shared_advs = {}, {}, {}, {}, {}
        
        for w in range(self.workers):
            self.last_episodes_completed = 0
            self.steps_during_last_gather = 0

            s_trajectory, r_trajectory, a_trajectory, aprob_trajectory, t_is_terminal = [], [], [], [], []
            timestep = 1
            current_episode_return = 0
            state = tf.cast(tf.reshape(self.env.reset(), [1, -1]), dtype=tf.float64)
            for t in range(self.horizon):
                self.total_frames += 1
                self.steps_during_last_gather += 1

                # choose action and step
                action, action_probability = agent.act(state)
                observation, reward, done, _ = self.env.step(
                    numpy.atleast_1d(action.numpy()) if self.env_is_continuous else action.numpy()
                )

                # remember experience
                s_trajectory.append(tf.cast(tf.reshape(state, [-1]), dtype=tf.float64))
                a_trajectory.append(action)
                aprob_trajectory.append(action_probability)
                r_trajectory.append(reward)
                t_is_terminal.append(done == 1)
                current_episode_return += reward

                # next state
                if done:
                    state = tf.cast(tf.reshape(self.env.reset(), [1, -1]), dtype=tf.float64)
                    self.episode_length_history.append(timestep)
                    self.episode_reward_history.append(current_episode_return)
                    self.last_episodes_completed += 1

                    timestep = 1
                    current_episode_return = 0
                else:
                    state = tf.cast(tf.reshape(observation, [1, -1]), dtype=tf.float64)
                    timestep += 1

            self.mean_episode_reward_per_gathering.append(0 if self.last_episodes_completed < 1 else statistics.mean(
                self.episode_reward_history[-self.last_episodes_completed:]))
            self.stdev_episode_reward_per_gathering.append(0 if self.last_episodes_completed <= 1 else statistics.stdev(
                self.episode_reward_history[-self.last_episodes_completed:]))

            # value prediction needs to include last state that was not included too, in order to make GAE possible
            s_trajectory.append(state)
            value_predictions = [agent.critic(tf.reshape(s, [1, -1]))[0][0] for s in s_trajectory]
            advantages = estimate_advantage(r_trajectory, value_predictions, t_is_terminal, gamma=agent.discount,
                                            lam=agent.lam)
            returns = numpy.add(advantages, value_predictions[:-1])
            advantages = self.normalize_advantages(advantages)

            # store this worker's gathering in the shared memory
            shared_s_trajectory[w] = copy.deepcopy(s_trajectory[:-1])
            shared_a_trajectory[w] = copy.deepcopy(a_trajectory)
            shared_aprob_trajectory[w] = copy.deepcopy(aprob_trajectory)
            shared_r[w] = copy.deepcopy(returns)
            shared_advs[w] = copy.deepcopy(advantages)

        # merge the shared memory
        joined_s_trajectory, joined_a_trajectory, joined_aprob_trajectory, joined_r, joined_advs = [], [], [], [], []
        for w_id in range(self.workers):
            joined_s_trajectory.extend(shared_s_trajectory[w_id])
            joined_a_trajectory.extend(shared_a_trajectory[w_id])
            joined_aprob_trajectory.extend(shared_aprob_trajectory[w_id])
            joined_r.extend(shared_r[w_id])
            joined_advs.extend(shared_advs[w_id])

        return tf.data.Dataset.from_tensor_slices({
            "state": joined_s_trajectory,
            "action": joined_a_trajectory,
            "action_prob": joined_aprob_trajectory,
            "return": joined_r,
            "advantage": joined_advs
        })
