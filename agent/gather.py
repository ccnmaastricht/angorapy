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
from models import build_ffn_models, build_rnn_models, GaussianPolicyDistribution
from utilities.const import STORAGE_DIR, DETERMINISTIC, COLORS, DEBUG
from utilities.datatypes import ExperienceBuffer, ModelTuple, TimeSequenceExperienceBuffer
from utilities.model_management import is_recurrent_model
from utilities.util import parse_state, add_state_dims, flatten, env_extract_dims
from utilities.wrappers import CombiWrapper, RewardNormalizationWrapper, StateNormalizationWrapper, BaseWrapper


@ray.remote(num_cpus=1, num_gpus=0, max_calls=10)
def collect(policy_tuple, horizon: int, env_name: str, discount: float, lam: float, subseq_length: int, pid: int,
            preprocessor_serialized: dict):
    """Collect a batch shard of experience for a given number of timesteps."""

    st = time.time()
    # import here to avoid pickling errors
    import tensorflow as tfl

    # build new environment for each collector to make multiprocessing possible
    env = gym.make(env_name)
    if DETERMINISTIC:
        env.seed(1)

    is_continuous = isinstance(env.action_space, Box)
    is_shadow_brain = "ShadowHand" in env_name

    preprocessor = BaseWrapper.from_serialization(preprocessor_serialized)

    # load policy
    if isinstance(policy_tuple, ModelTuple):
        model_builder = getattr(models, policy_tuple.model_builder)
        distribution = getattr(policies, policy_tuple.distribution_type)(env)

        # recurrent policy needs batch size for statefulness
        _, _, joint = model_builder(env, distribution, **({"bs": 1} if "bs" in fargs(model_builder).args else {}))
        joint.set_weights(policy_tuple.weights)
    else:
        raise ValueError("Unknown input for model.")

    # check if there is a recurrent layer inside the model
    is_recurrent = is_recurrent_model(joint)

    # buffer storing the experience and stats
    if is_recurrent:
        assert horizon % subseq_length == 0, "Subsequence length would require cutting of part of the observations."
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
    state = preprocessor.modulate((state, None, None, None))[0]

    setup_time = time.time() - st
    st = time.time()
    pure_choice_time, pure_act_time, pure_sim_time, pure_norm_time, pure_eps_end_time = 0, 0, 0, 0, 0

    while t < horizon:
        current_subseq_length += 1

        # based on the given state, predict action distribution and state value; need flatten due to tf eager bug
        stt = time.time()

        policy_out = flatten(joint.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))
        pure_choice_time += time.time() - stt
        a_distr, value = policy_out[:-1], policy_out[-1]
        states.append(state)
        values.append(np.squeeze(value))

        # from the action distribution sample an action and remember both the action and its probability
        stt = time.time()
        action, action_probability = distribution.act(*a_distr)
        pure_act_time += time.time() - stt

        action = action if not DETERMINISTIC else np.zeros(action.shape)
        actions.append(action)
        action_probabilities.append(action_probability)  # should probably ensure that no probability is ever 0

        # make a step based on the chosen action and collect the reward for this state
        stt = time.time()
        observation, reward, done, _ = env.step(np.atleast_1d(action) if is_continuous else action)
        pure_sim_time += time.time() - stt
        current_episode_return += reward  # true reward for stats
        stt = time.time()
        observation, reward, done, _ = preprocessor.modulate((parse_state(observation), reward, done, None))
        pure_norm_time += time.time() - stt

        rewards.append(reward)

        stt = time.time()
        # if recurrent, at a subsequence breakpoint or episode end stack the observations and give them to the buffer
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
            episode_advantages = estimate_episode_advantages(rewards[-episode_steps:], values[-episode_steps:] + [0],
                                                             discount, lam)
            episode_returns = episode_advantages + values[-episode_steps:]

            if not is_recurrent:
                advantages.append(episode_advantages)
            else:
                buffer.push_adv_ret_to_buffer(episode_advantages, episode_returns)

                # skip as many steps as are missing to fill the subsequence, then reset rnn states for next episode
                t += subseq_length - (t % subseq_length) - 1
                joint.reset_states()

            # reset environment to receive next episodes initial state
            state = parse_state(env.reset())
            state = preprocessor.modulate((state, None, None, None))[0]

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

    env.close()

    stepping_time = time.time() - st
    st = time.time()

    # get last non-visited state's value to incorporate it into the advantage estimation of last visited state
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

    if DEBUG:
        mc = COLORS["FAIL"]
        ec = COLORS["ENDC"]
        np.set_printoptions(linewidth=250, floatmode="fixed")
        print("LAST FIVES\n---------")
        print(f"{mc}states{ec}: {buffer.states[-5:]}")
        print(f"{mc}rewards{ec}: {rewards[-5:]}")
        print(f"{mc}advantages{ec}: {advantages[-5:]}")
        print(f"{mc}returns{ec}: {returns[-5:]}")
        print(f"{mc}values{ec}: {buffer.values[-5:]}")
        print(f"{mc}actions{ec}: {buffer.actions[-5:]}")
        print(f"{mc}action probabilities{ec}: {buffer.action_probabilities[-5:]}")
        print(f"{mc}action mean{ec}: {np.mean(actions)}: {np.mean(actions, axis=0)}")
        print(f"{mc}values mean{ec}: {np.mean(buffer.values)}")

        print("\n\nWRAPPERS\n---------")
        print(f"{mc}State Wrapper Mean{ec}: {preprocessor_serialized[0].mean}")
        print(f"{mc}State Wrapper Var{ec}: {preprocessor_serialized[0].variance}")
        print(f"{mc}Reward Wrapper Mean{ec}: {preprocessor_serialized[1].mean}")
        print(f"{mc}Reward Wrapper Variance{ec}: {preprocessor_serialized[1].variance}")

        import matplotlib.pyplot as plt
        import matplotlib
        plt.hist(buffer.values, label="value")
        plt.hist(buffer.returns, label="return")
        plt.hist(advantages, label="adv")
        plt.hist(rewards, label="rewards")
        plt.hist(tf.square(buffer.values - buffer.returns), label="squared value-return")
        plt.hist(buffer.action_probabilities, label="aprob")
        plt.hist(np.mean(buffer.actions, axis=-1), label="action")

        # plt.hist(buffer.advantages + buffer.values, label="return after norm")
        matplotlib.use('TkAgg')
        plt.legend()
        plt.title("GATHER DISTRIBUTION")
        plt.show()
        exit()

    if is_recurrent:
        # add batch dimension for optimization
        buffer.inject_batch_dimension()

    wrapup_time = time.time() - st
    st = time.time()

    # convert buffer to dataset and save it to tf record
    dataset, stats = make_dataset_and_stats(buffer, is_shadow_brain=is_shadow_brain)
    dataset = dataset.map(tf_serialize_example)

    # TODO I have the suspicion that the writer leaks memory if we wouldn't reset the workers
    writer = tfl.data.experimental.TFRecordWriter(f"{STORAGE_DIR}/data_{pid}.tfrecord")
    writer.write(dataset)

    savenwrite_time = time.time() - st

    # if pid == 0:
    #     print(f"\n\nTotal Runtime: {sum([setup_time, stepping_time, wrapup_time, savenwrite_time])}")
    #     print(
    #         f"\tSetup: {setup_time}\n\tStep: {stepping_time}\n\t\tChoice: {pure_choice_time}\n\t\tAct: {pure_act_time}"
    #         f"\n\t\tSim: {pure_sim_time}\n\t\tNorm: {pure_norm_time}\n\t\tEnding: {pure_eps_end_time}"
    #         f"\n\tWrapup: {wrapup_time}\n\tSaving: {savenwrite_time}")

    return stats, preprocessor


@ray.remote(num_cpus=1, num_gpus=0)
def evaluate(policy_tuple, env_name: str, preprocessor_serialized: dict) -> Tuple[int, int]:
    """Evaluate one episode of the given environment following the given policy. Remote implementation."""
    env = gym.make(env_name)
    preprocessor = BaseWrapper.from_serialization(preprocessor_serialized)

    if isinstance(policy_tuple, ModelTuple):
        model_builder = getattr(models, policy_tuple.model_builder)
        distribution = getattr(policies, policy_tuple.distribution_type)(env)

        # recurrent policy needs batch size for statefulness
        policy, _, _ = model_builder(env, distribution, **({"bs": 1} if "bs" in fargs(model_builder).args else {}))
        policy.set_weights(policy_tuple.weights)
    else:
        raise ValueError("Cannot handle given model type. Should be a ModelTuple.")

    is_recurrent = is_recurrent_model(policy)
    done = False
    reward_trajectory = []
    length = 0
    state = env.reset()
    state = preprocessor.modulate((state, None, None, None), update=False)[0]
    while not done:
        probabilities = flatten(policy.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))

        action, _ = distribution.act(*probabilities)
        observation, reward, done, _ = env.step(action)
        reward_trajectory.append(reward)
        state, reward, done, _ = preprocessor.modulate((observation, reward, done, None), update=False)
        length += 1

    return length, sum(reward_trajectory)


if __name__ == "__main__":
    """Performance Measuring."""

    RUN_DEBUG = False

    os.chdir("../")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # need to deactivate GPU in Debugger mode!
    tf.config.experimental_run_functions_eagerly(RUN_DEBUG)

    env_n = "HalfCheetah-v2"
    env = gym.make(env_n)
    distribution = GaussianPolicyDistribution(env)
    sd, ad = env_extract_dims(env)
    p, v, j = models.get_model_builder("ffn", False)(env, distribution)
    joint_tuple = ModelTuple(build_ffn_models.__name__, j.get_weights(), distribution.__class__.__name__)
    rp, rv, rj = models.get_model_builder("rnn", False)(env, distribution)
    rjoint_tuple = ModelTuple(build_rnn_models.__name__, rj.get_weights(), distribution.__class__.__name__)

    wrapper = CombiWrapper((StateNormalizationWrapper(sd), RewardNormalizationWrapper()))

    ray.init()
    t = time.time()
    rms_ffn = [collect.remote(joint_tuple, 2048, env_n, 0.99, 0.95, 16, i, wrapper.serialize()) for i in range(1)]
    outs_ffn = [ray.get(rm) for rm in rms_ffn]
    print(f"Programm Runtime: {time.time() - t}")

    # remote function, 8 workers, 2048 horizon: Programm Runtime: 24.98351287841797
    # remote function, 1 worker, 2048 horizon: Programm Runtime: 10.563997030258179

