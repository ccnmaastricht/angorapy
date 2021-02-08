#!/usr/bin/env python
"""Functions for gathering experience and communicating it to the main thread."""

import time
from inspect import getfullargspec as fargs
from typing import Tuple, Any, Union

import numpy as np
import tensorflow as tf
from gym.spaces import Box
from mpi4py import MPI

import models
from agent import policies
from agent.core import estimate_episode_advantages
from agent.dataio import tf_serialize_example, make_dataset_and_stats
from agent.policies import GaussianPolicyDistribution, BasePolicyDistribution
from common.wrappers import make_env, BaseWrapper
from models import build_ffn_models
from utilities.const import STORAGE_DIR, DETERMINISTIC
from utilities.datatypes import ExperienceBuffer, TimeSequenceExperienceBuffer, StatBundle
from utilities.model_utils import is_recurrent_model
from utilities.util import parse_state, add_state_dims, flatten, env_extract_dims


class Gatherer:
    """Remote Gathering class."""

    def __init__(self, worker_id: int, exp_id: int):
        # worker identification
        self.worker_id = worker_id
        self.exp_id = exp_id

    def collect(self, env: BaseWrapper, joint: tf.keras.Model, distribution: BasePolicyDistribution,
                horizon: int, discount: float, lam: float, subseq_length: int, collector_id: int) -> StatBundle:
        """Collect a batch shard of experience for a given number of timesteps.

        Args:
            env:            environment from which to gather the data
            joint:          joint model of the policy, having both an action and a value head
            distribution:   policy distribution object
            horizon:        the number of steps gatherd by this worker
            discount:       discount factor
            lam:            lambda parameter of GAE balancing the tradeoff between bias and variance
            subseq_length:  the length of connected subsequences for TBPTT
            collector_id:   the ID of this gathering, different from the worker's ID
        """
        is_recurrent = is_recurrent_model(joint)
        is_continuous = isinstance(env.action_space, Box)
        is_shadow_brain = "BaseShadowHand" in env.unwrapped.spec.id

        if DETERMINISTIC:
            env.seed(1)

        # reset states of potentially recurrent net
        joint.reset_states()

        # buffer storing the experience and stats
        if is_recurrent:
            assert horizon % subseq_length == 0, "Subsequence length would require cutting of part of the n_steps."
            buffer: TimeSequenceExperienceBuffer = TimeSequenceExperienceBuffer.new(env=env,
                                                                                    size=horizon // subseq_length,
                                                                                    seq_len=subseq_length,
                                                                                    is_continuous=is_continuous,
                                                                                    is_multi_feature=is_shadow_brain)
        else:
            buffer: ExperienceBuffer = ExperienceBuffer.new_empty(is_continuous, is_shadow_brain)

        # go for it
        t, current_episode_return, episode_steps, current_subseq_length = 0, 0, 1, 0
        states, rewards, actions, action_probabilities, values, advantages = [], [], [], [], [], []
        episode_endpoints = []
        state = parse_state(env.reset())
        while t < horizon:
            current_subseq_length += 1

            # based on given state, predict step_tuple distribution and state value; need flatten due to tf eager bug
            policy_out = flatten(
                joint.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))
            a_distr, value = policy_out[:-1], policy_out[-1]
            states.append(state)
            values.append(np.squeeze(value))

            # from the step_tuple distribution sample an step_tuple and remember both the step_tuple and its probability
            action, action_probability = distribution.act(*a_distr)

            action = action if not DETERMINISTIC else np.zeros(action.shape)
            actions.append(action)
            action_probabilities.append(action_probability)  # should probably ensure that no probability is ever 0

            # make a step based on the chosen step_tuple and collect the reward for this state
            observation, reward, done, _ = env.step(np.atleast_1d(action) if is_continuous else action)
            current_episode_return += reward  # true reward for stats

            observation = parse_state(observation)
            rewards.append(reward)

            # if recurrent, at a subsequence breakpoint/episode end stack the n_steps and buffer them
            if is_recurrent and (current_subseq_length == subseq_length or done):
                buffer.push_seq_to_buffer(states, actions, action_probabilities, values[-current_subseq_length:])

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

                if not is_recurrent:
                    advantages.append(episode_advantages)
                else:
                    # skip as many steps as are missing to fill the subsequence, then push adv ant ret to buffer
                    t += subseq_length - (t % subseq_length) - 1
                    buffer.push_adv_ret_to_buffer(episode_advantages, episode_returns)

                # reset environment to receive next episodes initial state
                state = parse_state(env.reset())
                joint.reset_states()

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

        # get last non-visited state'serialization value to incorporate it into the advantage estimation of last visited state
        values.append(np.squeeze(joint.predict(add_state_dims(state, dims=2 if is_recurrent else 1))[-1]))

        # if there was at least one step in the environment after the last episode end, calculate advantages for them
        if episode_steps > 1:
            leftover_advantages = estimate_episode_advantages(rewards[-episode_steps + 1:], values[-episode_steps:],
                                                              discount, lam)
            if not is_recurrent:
                advantages.append(leftover_advantages)
            else:
                leftover_returns = leftover_advantages + values[-len(leftover_advantages) - 1:-1]
                buffer.push_adv_ret_to_buffer(leftover_advantages, leftover_returns)

        # if not recurrent, fill the buffer with everything we gathered
        if not is_recurrent:
            values = np.array(values, dtype="float32")

            # write to the buffer
            advantages = np.hstack(advantages).astype("float32")
            returns = advantages + values[:-1]
            buffer.fill(np.array(states, dtype="float32"),
                        np.array(actions, dtype="float32" if is_continuous else "int32"),
                        np.array(action_probabilities, dtype="float32"),
                        advantages,
                        returns,
                        values[:-1])

        # normalize advantages
        buffer.normalize_advantages()

        if is_recurrent:
            buffer.inject_batch_dimension()

        # convert buffer to dataset and save it to tf record
        dataset, stats = make_dataset_and_stats(buffer, is_shadow_brain=is_shadow_brain)
        dataset = dataset.map(tf_serialize_example)

        writer = tf.data.experimental.TFRecordWriter(f"{STORAGE_DIR}/{self.exp_id}_data_{collector_id}.tfrecord")
        writer.write(dataset)

        return stats

    def evaluate(self, env: BaseWrapper, policy: tf.keras.Model, distribution: BasePolicyDistribution) -> \
            Tuple[int, int, Any]:
        """Evaluate one episode of the given environment following the given policy. Remote implementation."""

        # reset policy states as it might be recurrent
        policy.reset_states()

        done = False
        state = parse_state(env.reset())
        cumulative_reward = 0
        steps = 0
        while not done:
            probabilities = flatten(
                policy.predict(add_state_dims(parse_state(state), dims=2 if self.is_recurrent else 1)))

            action, _ = distribution.act(*probabilities)
            observation, reward, done, _ = env.step(action)
            cumulative_reward += reward
            observation = parse_state(observation)

            state = observation
            steps += 1

        eps_class = env.unwrapped.current_target_finger if hasattr(env.unwrapped, "current_target_finger") else None

        return steps, cumulative_reward, eps_class


if __name__ == "__main__":
    """Performance Measuring."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env_n = "HalfCheetah-v2"
    environment = make_env(env_n)
    distro = GaussianPolicyDistribution(environment)
    builder = build_ffn_models
    sd, ad = env_extract_dims(environment)
    wrapper = CombiWrapper((StateNormalizationWrapper(sd), RewardNormalizationWrapper()))

    n_actors = 8
    base, extra = divmod(n_actors, size)
    n_actors_on_this_node = base + (rank < extra)
    print(n_actors_on_this_node)

    t = time.time()
    actors = [Gatherer(i, ) for i in range(n_actors_on_this_node)]

    it = time.time()
    outs_ffn = [actor.collect(512, 0.99, 0.95, 16, ) for actor in actors]
    gathering_msg = f"Gathering Time: {time.time() - it}"

    msgs = comm.gather(gathering_msg, root=0)

    if rank == 0:
        for msg in msgs:
            print(msg)

    print(f"Program Runtime: {time.time() - t}")

    # remote function, 8 workers, 2048 horizon: Program Runtime: 24.98351287841797
    # remote function, 1 worker, 2048 horizon: Program Runtime: 10.563997030258179
