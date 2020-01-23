#!/usr/bin/env python
"""Functions for gathering experience and communicating it to the main thread."""
import os
import time
from inspect import getfullargspec as fargs
from typing import Tuple

import numpy as np
import ray
import tensorflow as tf
from gym.spaces import Box

import models
from agent import policies
from agent.core import estimate_episode_advantages
from agent.dataio import tf_serialize_example, make_dataset_and_stats
from environments import *
from models import build_rnn_models, GaussianPolicyDistribution
from utilities.const import STORAGE_DIR, DETERMINISTIC
from utilities.datatypes import ExperienceBuffer, TimeSequenceExperienceBuffer
from utilities.model_utils import is_recurrent_model
from utilities.util import parse_state, add_state_dims, flatten, env_extract_dims
from utilities.wrappers import CombiWrapper, RewardNormalizationWrapper, StateNormalizationWrapper, BaseWrapper


@ray.remote
class Gatherer:
    """Remote Gathering class."""

    def __init__(self, model_builder_name: str, distribution_name: str, env_name: str, worker_id: int):
        model_builder = getattr(models, model_builder_name)

        self.id = worker_id

        # setup persistent tools
        self.env = gym.make(env_name)
        self.distribution = getattr(policies, distribution_name)(self.env)
        self.policy, _, self.joint = model_builder(
            self.env, self.distribution, **({"bs": 1} if "bs" in fargs(model_builder).args else {}))

        # some attributes for adaptive behaviour
        self.is_recurrent = is_recurrent_model(self.joint)
        self.is_continuous = isinstance(self.env.action_space, Box)
        self.is_shadow_brain = "ShadowHand" in env_name

    def update_weights(self, weights):
        """Update the weights of this worker."""
        self.joint.set_weights(weights)

    def collect(self, horizon: int, discount: float, lam: float, subseq_length: int, preprocessor_serialized: dict,
                verbose: bool = False):
        """Collect a batch shard of experience for a given number of timesteps."""

        st = time.time()
        # import here to avoid pickling errors
        import tensorflow as tfl

        # build new environment for each collector to make multiprocessing possible
        if DETERMINISTIC:
            self.env.seed(1)

        preprocessor = BaseWrapper.from_serialization(preprocessor_serialized)

        # buffer storing the experience and stats
        if self.is_recurrent:
            assert horizon % subseq_length == 0, "Subsequence length would require cutting of part of the observations."
            buffer: TimeSequenceExperienceBuffer = TimeSequenceExperienceBuffer.new(env=self.env,
                                                                                    size=horizon // subseq_length,
                                                                                    seq_len=subseq_length,
                                                                                    is_continuous=self.is_continuous,
                                                                                    is_multi_feature=self.is_shadow_brain)
        else:
            buffer: ExperienceBuffer = ExperienceBuffer.new_empty(self.is_continuous, self.is_shadow_brain)

        # go for it
        t, current_episode_return, episode_steps, current_subseq_length = 0, 0, 1, 0
        states, rewards, actions, action_probabilities, values, advantages = [], [], [], [], [], []
        episode_endpoints = []
        state = preprocessor.modulate((parse_state(self.env.reset()), None, None, None))[0]

        setup_time, st = time.time() - st, time.time()
        pure_choice_time, pure_act_time, pure_sim_time, pure_norm_time, pure_eps_end_time = 0, 0, 0, 0, 0

        while t < horizon:
            current_subseq_length += 1

            # based on the given state, predict action distribution and state value; need flatten due to tf eager bug
            stt = time.time()

            policy_out = flatten(
                self.joint.predict(add_state_dims(parse_state(state), dims=2 if self.is_recurrent else 1)))
            pure_choice_time += time.time() - stt
            a_distr, value = policy_out[:-1], policy_out[-1]
            states.append(state)
            values.append(np.squeeze(value))

            # from the action distribution sample an action and remember both the action and its probability
            stt = time.time()
            action, action_probability = self.distribution.act(*a_distr)
            pure_act_time += time.time() - stt

            action = action if not DETERMINISTIC else np.zeros(action.shape)
            actions.append(action)
            action_probabilities.append(action_probability)  # should probably ensure that no probability is ever 0

            # make a step based on the chosen action and collect the reward for this state
            stt = time.time()
            observation, reward, done, _ = self.env.step(np.atleast_1d(action) if self.is_continuous else action)
            pure_sim_time += time.time() - stt
            current_episode_return += reward  # true reward for stats
            stt = time.time()
            observation, reward, done, _ = preprocessor.modulate((parse_state(observation), reward, done, None))
            pure_norm_time += time.time() - stt

            rewards.append(reward)

            stt = time.time()
            # if recurrent, at a subsequence breakpoint or episode end stack the observations and give them to the buffer
            if self.is_recurrent and (current_subseq_length == subseq_length or done):
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

                if not self.is_recurrent:
                    advantages.append(episode_advantages)
                else:
                    buffer.push_adv_ret_to_buffer(episode_advantages, episode_returns)

                    # skip as many steps as are missing to fill the subsequence, then reset rnn states for next episode
                    t += subseq_length - (t % subseq_length) - 1
                    self.joint.reset_states()

                # reset environment to receive next episodes initial state
                state = preprocessor.modulate((parse_state(self.env.reset()), None, None, None))[0]

                # update/reset some statistics and trackers
                buffer.episode_lengths.append(episode_steps)
                buffer.episode_rewards.append(current_episode_return)
                buffer.episodes_completed += 1
                episode_steps = 1
                current_episode_return = 0
            else:
                state = observation
                episode_steps += 1
            pure_eps_end_time += time.time() - stt

            t += 1

        self.env.close()

        stepping_time, st = time.time() - st, time.time()

        # get last non-visited state's value to incorporate it into the advantage estimation of last visited state
        values.append(np.squeeze(self.joint.predict(add_state_dims(state, dims=2 if self.is_recurrent else 1))[-1]))

        # if there was at least one step in the environment after the last episode end, calculate advantages for them
        if episode_steps > 1:
            leftover_advantages = estimate_episode_advantages(rewards[-episode_steps + 1:], values[-episode_steps:],
                                                              discount, lam)
            if not self.is_recurrent:
                advantages.append(leftover_advantages)
            else:
                leftover_returns = leftover_advantages + values[-len(leftover_advantages) - 1:-1]
                buffer.push_adv_ret_to_buffer(leftover_advantages, leftover_returns)

        # if not recurrent, fill the buffer with everything we gathered
        if not self.is_recurrent:
            values = np.array(values, dtype="float32")

            # write to the buffer
            advantages = np.hstack(advantages).astype("float32")
            returns = advantages + values[:-1]
            buffer.fill(np.array(states, dtype="float32"),
                        np.array(actions, dtype="float32" if self.is_continuous else "int32"),
                        np.array(action_probabilities, dtype="float32"),
                        advantages,
                        returns,
                        values[:-1])

        # normalize advantages
        buffer.normalize_advantages()

        if self.is_recurrent:
            buffer.inject_batch_dimension()

        wrapup_time, st = time.time() - st, time.time()

        # convert buffer to dataset and save it to tf record
        dataset, stats = make_dataset_and_stats(buffer, is_shadow_brain=self.is_shadow_brain)
        dataset = dataset.map(tf_serialize_example)

        # TODO I have the suspicion that the writer leaks memory if we wouldn't reset the workers
        writer = tfl.data.experimental.TFRecordWriter(f"{STORAGE_DIR}/data_{self.id}.tfrecord")
        writer.write(dataset)

        savenwrite_time = time.time() - st

        if self.id == 0 and verbose:
            print(f"\n\nTotal Runtime: {sum([setup_time, stepping_time, wrapup_time, savenwrite_time])}")
            print(
                f"\tSetup: {setup_time}\n\tStep: {stepping_time}\n\t\tChoice: {pure_choice_time}\n\t\tAct: {pure_act_time}"
                f"\n\t\tSim: {pure_sim_time}\n\t\tNorm: {pure_norm_time}\n\t\tEnding: {pure_eps_end_time}"
                f"\n\tWrapup: {wrapup_time}\n\tSaving: {savenwrite_time}")

        return stats, preprocessor

    def evaluate(self, preprocessor_serialized: dict) -> Tuple[int, int]:
        """Evaluate one episode of the given environment following the given policy. Remote implementation."""
        preprocessor = BaseWrapper.from_serialization(preprocessor_serialized)

        done = False
        state = preprocessor.modulate((parse_state(self.env.reset()), None, None, None), update=False)[0]
        cumulative_reward = 0
        steps = 0
        while not done:
            probabilities = flatten(
                self.policy.predict(add_state_dims(parse_state(state), dims=2 if self.is_recurrent else 1)))

            action, _ = self.distribution.act(*probabilities)
            observation, reward, done, _ = self.env.step(action)
            cumulative_reward += reward
            observation, reward, done, _ = preprocessor.modulate((parse_state(observation), reward, done, None),
                                                                 update=False)

            state = observation
            steps += 1

        return steps, cumulative_reward


if __name__ == "__main__":
    """Performance Measuring."""

    RUN_DEBUG = False

    os.chdir("../")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # need to deactivate GPU in Debugger mode!
    tf.config.experimental_run_functions_eagerly(RUN_DEBUG)

    env_n = "HalfCheetah-v2"
    environment = gym.make(env_n)
    distro = GaussianPolicyDistribution(environment)
    builder = build_rnn_models
    sd, ad = env_extract_dims(environment)
    wrapper = CombiWrapper((StateNormalizationWrapper(sd), RewardNormalizationWrapper()))

    ray.init()
    t = time.time()
    actors = [Gatherer.remote(builder.__name__, distro.__class__.__name__, env_n, i) for i in range(8)]

    for _ in range(100):
        it = time.time()
        outs_ffn = ray.get([actor.collect.remote(2048, 0.99, 0.95, 16, wrapper.serialize(), False) for actor in actors])
        print(f"Gathering Time: {time.time() - it}")

    print(f"Program Runtime: {time.time() - t}")

    # remote function, 8 workers, 2048 horizon: Program Runtime: 24.98351287841797
    # remote function, 1 worker, 2048 horizon: Program Runtime: 10.563997030258179
