#!/usr/bin/env python
"""Functions for gathering experience and communicating it to the main thread."""
import gc
from typing import Tuple, Any

import numpy as np
import tensorflow as tf
from gym.spaces import Box

from agent.core import estimate_episode_advantages
from agent.dataio import tf_serialize_example, make_dataset_and_stats
from common.data_buffers import ExperienceBuffer, TimeSequenceExperienceBuffer
from common.policies import BasePolicyDistribution
from common.senses import Sensation
from common.wrappers import BaseWrapper
from common.const import STORAGE_DIR, DETERMINISTIC
from utilities.datatypes import StatBundle
from utilities.model_utils import is_recurrent_model
from utilities.util import add_state_dims, flatten, env_extract_dims


class Gatherer:
    """Worker implementation for collecting experience by rolling out a policy."""

    def __init__(self, joint: tf.keras.Model, policy: tf.keras.Model, worker_id: int, exp_id: int):
        self.joint: tf.keras.Model = joint
        self.policy = policy
        self.worker_id = worker_id
        self.exp_id = exp_id
        self.iteration = 0

    def set_weights(self, weights):
        """Set the weights of the full model."""
        self.joint.set_weights(weights)

    def collect(self, env: BaseWrapper, distribution: BasePolicyDistribution, horizon: int, discount: float, lam: float,
                subseq_length: int, collector_id: int) -> StatBundle:
        """Collect a batch shard of experience for a given number of timesteps.

        Args:
            env:            environment from which to gather the data
            distribution:   policy distribution object
            horizon:        the number of steps gatherd by this worker
            discount:       discount factor
            lam:            lambda parameter of GAE balancing the tradeoff between bias and variance
            subseq_length:  the length of connected subsequences for TBPTT
            collector_id:   the ID of this gathering, different from the worker's ID
        """
        self.iteration += 1

        state: Sensation

        is_recurrent = is_recurrent_model(self.joint)
        is_continuous = isinstance(env.action_space, Box)
        state_dim, action_dim = env_extract_dims(env)

        if DETERMINISTIC:
            env.seed(1)

        # reset states of potentially recurrent net
        if is_recurrent:
            self.joint.reset_states()
            self.policy.reset_states()

        # buffer storing the experience and stats
        if is_recurrent:
            assert horizon % subseq_length == 0, "Subsequence length would require cutting of part of the n_steps."
            buffer = TimeSequenceExperienceBuffer(horizon // subseq_length, state_dim, action_dim,
                                                  is_continuous, subseq_length)
        else:
            buffer = ExperienceBuffer(horizon, state_dim, action_dim, is_continuous)

        # go for it
        t, current_episode_return, episode_steps, current_subseq_length = 0, 0, 1, 0
        states, rewards, actions, action_probabilities, values, advantages = [], [], [], [], [], []
        episode_endpoints = []
        state = env.reset()
        while t < horizon:
            current_subseq_length += 1

            # based on given state, predict action distribution and state value; need flatten due to tf eager bug
            prepared_state = state.with_leading_dims(time=is_recurrent).dict_as_tf()
            policy_out = flatten(self.joint(prepared_state))

            a_distr, value = policy_out[:-1], policy_out[-1]

            states.append(state)
            values.append(np.squeeze(value))

            # from the action distribution sample an action and remember both the action and its probability
            action, action_probability = distribution.act(*a_distr)
            action = action if not DETERMINISTIC else np.zeros(action.shape)

            actions.append(action)
            action_probabilities.append(action_probability)  # should probably ensure that no probability is ever 0

            # make a step based on the chosen action and collect the reward for this state
            observation, reward, done, info = env.step(np.atleast_1d(action) if is_continuous else action)
            current_episode_return += (reward if "original_reward" not in info else info["original_reward"])
            rewards.append(reward)

            # if recurrent, at a subsequence breakpoint/episode end stack the n_steps and buffer them
            if is_recurrent and (current_subseq_length == subseq_length or done):
                buffer.push_seq_to_buffer(states=states,
                                          actions=actions,
                                          action_probabilities=action_probabilities,
                                          values=values[-current_subseq_length:],
                                          episode_ended=done)

                # clear the buffered information
                states, actions, action_probabilities = [], [], []
                current_subseq_length = 0

            # depending on whether the state is terminal, choose the next state
            if done:
                episode_endpoints.append(t)

                # calculate advantages for the finished episode, where the last value is 0 since it refers to the
                # terminal state that we just observed
                episode_advantages = estimate_episode_advantages(rewards[-episode_steps:],
                                                                 values[-episode_steps:] + [0],
                                                                 discount, lam)
                episode_returns = episode_advantages + values[-episode_steps:]

                if is_recurrent:
                    # skip as many steps as are missing to fill the subsequence, then push adv ant ret to buffer
                    t += subseq_length - (t % subseq_length) - 1
                    buffer.push_adv_ret_to_buffer(episode_advantages, episode_returns)
                else:
                    advantages.append(episode_advantages)

                # reset environment to receive next episodes initial state
                state = env.reset()

                if is_recurrent:
                    self.joint.reset_states()
                    self.policy.reset_states()

                # update/reset some statistics and trackers
                buffer.episode_lengths.append(episode_steps)
                buffer.episode_rewards.append(current_episode_return)
                buffer.episodes_completed += 1
                episode_steps = 1
                current_episode_return = 0
            else:
                state = observation
                episode_steps += 1

            t += 1

        env.close()

        # get last non-visited state value to incorporate it into the advantage estimation of last visited state
        values.append(np.squeeze(self.joint(add_state_dims(state, dims=2 if is_recurrent else 1).dict())[-1]))

        # if there was at least one step in the environment after the last episode end, calculate advantages for them
        if episode_steps > 1:
            leftover_advantages = estimate_episode_advantages(rewards[-episode_steps + 1:], values[-episode_steps:],
                                                              discount, lam)
            if is_recurrent:
                leftover_returns = leftover_advantages + values[-len(leftover_advantages) - 1:-1]
                buffer.push_adv_ret_to_buffer(leftover_advantages, leftover_returns)
            else:
                advantages.append(leftover_advantages)

        # if not recurrent, fill the buffer with everything we gathered
        if not is_recurrent:
            values = np.array(values, dtype="float32")

            # write to the buffer
            advantages = np.hstack(advantages).astype("float32")
            returns = advantages + values[:-1]
            buffer.fill(states,
                        np.array(actions, dtype="float32" if is_continuous else "int32"),
                        np.array(action_probabilities, dtype="float32"),
                        advantages,
                        returns,
                        values[:-1])

        # normalize advantages
        buffer.normalize_advantages()

        # convert buffer to dataset and save it to tf record
        dataset, stats = make_dataset_and_stats(buffer)
        dataset = dataset.map(tf_serialize_example)

        writer = tf.data.experimental.TFRecordWriter(f"{STORAGE_DIR}/{self.exp_id}_data_{collector_id}.tfrecord")
        writer.write(dataset)

        return stats

    def evaluate(self, env: BaseWrapper, distribution: BasePolicyDistribution) -> \
            Tuple[int, int, Any]:
        """Evaluate one episode of the given environment following the given policy. Remote implementation."""

        # reset policy states as it might be recurrent
        self.policy.reset_states()

        done = False
        state = env.reset()
        cumulative_reward = 0
        steps = 0
        while not done:
            probabilities = flatten(
                self.policy(add_state_dims(state, dims=2 if self.is_recurrent else 1)))

            action, _ = distribution.act(*probabilities)
            observation, reward, done, _ = env.step(action)
            cumulative_reward += reward
            observation = observation

            state = observation
            steps += 1

        eps_class = env.unwrapped.current_target_finger if hasattr(env.unwrapped, "current_target_finger") else None

        return steps, cumulative_reward, eps_class
