#!/usr/bin/env python
"""Functions for gathering experience and communicating it to the main thread."""
import os
from inspect import getfullargspec as fargs
from typing import Tuple

import numpy as np
import ray
from gym.spaces import Box, Discrete
from tqdm import tqdm

import models
from agent.core import estimate_episode_advantages
from agent.dataio import tf_serialize_example, make_dataset_and_stats
from agent.policy import act_discrete, act_continuous
from environments import *
from models import build_rnn_distinct_models, build_ffn_distinct_models
from utilities.const import STORAGE_DIR
from utilities.datatypes import ExperienceBuffer, ModelTuple
from utilities.util import parse_state, add_state_dims, is_recurrent_model


@ray.remote(num_cpus=1, num_gpus=0)
def collect(model, horizon: int, env_name: str, discount: float, lam: float, subseq_length: int, pid: int):
    """Collect a batch shard of experience for a given number of timesteps."""

    # import here to avoid pickling errors
    import tensorflow as tfl

    # build new environment for each collector to make multiprocessing possible
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, Box)
    act = act_continuous if is_continuous else act_discrete

    # load policy
    if isinstance(model, str):
        joint = tfl.keras.models.load_model(f"{model}/model")
    elif isinstance(model, ModelTuple):
        model_builder = getattr(models, model.model_builder)

        # recurrent policy needs batch size for statefulness
        _, _, joint = model_builder(env, **({"bs": 1} if "bs" in fargs(model_builder).args else {}))
        joint.set_weights(model.weights)
    else:
        raise ValueError("Unknown input for model.")

    # check if there is a recurrent layer inside the model
    is_recurrent = is_recurrent_model(joint)
    if is_recurrent:
        assert horizon % subseq_length == 0, "Subsequence length for TBPTT would require cutting of part of the" \
                                             " observations."
    is_shadow_brain = "ShadowHand" in env_name

    # buffer storing the experience and stats
    buffer: ExperienceBuffer = ExperienceBuffer.new_empty()

    # go for it
    t, current_episode_return, episode_steps, current_subseq_length = 0, 0, 1, 0
    states, rewards, actions, action_probabilities, values, advantages = [], [], [], [], [], []
    state = parse_state(env.reset())
    while t < horizon:
        current_subseq_length += 1

        # based on the given state, predict action distribution and state value
        action_distribution, value = joint.predict(add_state_dims(state, dims=2 if is_recurrent else 1))
        states.append(state)
        values.append(np.squeeze(value))

        # from the action distribution sample an action and remember both the action and its probability
        action, action_probability = act(action_distribution)
        actions.append(action)
        action_probabilities.append(action_probability)  # should probably ensure that no probability is ever 0

        # make a step based on the chosen action and collect the reward for this state
        observation, reward, done, _ = env.step(np.atleast_1d(action) if is_continuous else action)
        rewards.append(reward)
        current_episode_return += reward

        # if recurrent, at a subsequence breakpoint or episode end stack the observations and give them to the buffer
        if is_recurrent and (current_subseq_length == subseq_length or done):
            subseq_advantages = estimate_episode_advantages(rewards[-current_subseq_length:],
                                                            values[-current_subseq_length:] + [0],  # TODO proper last v
                                                            discount, lam)
            subseq_returns = subseq_advantages + values[-current_subseq_length:]
            buffer.push_seq_to_buffer(states, actions, action_probabilities, subseq_advantages, subseq_returns,
                                      is_multi_feature=is_shadow_brain, is_continuous=is_continuous)

            # reset the buffered information
            states, actions, action_probabilities = [], [], []
            current_subseq_length = 0

        # depending on whether the state is terminal, choose the next state
        if done:
            if not is_recurrent:
                # calculate advantages for the finished episode, where the last value is 0 since it refers to the
                # terminal state that we just observed
                advantages.append(estimate_episode_advantages(rewards[-episode_steps:],
                                                              values[-episode_steps:] + [0],
                                                              discount, lam))
            else:
                # skip as many steps as are missing to fill the subsequence, then reset rnn states for next episode
                t += (subseq_length - (t % subseq_length)) - 1
                joint.reset_states()

            # reset environment to receive next episodes initial state
            state = parse_state(env.reset())

            # update/reset some statistics and trackers
            buffer.episode_lengths.append(episode_steps)
            buffer.episode_rewards.append(current_episode_return)
            buffer.episodes_completed += 1
            episode_steps = 1
            current_episode_return = 0
        else:
            state = parse_state(observation)
            episode_steps += 1

        t += 1

    env.close()

    # non-recurrent and recurrent wrap up
    if not is_recurrent:
        # get last non-visited state's value to incorporate it into the advantage estimation of last visited state
        values.append(np.squeeze(joint.predict(add_state_dims(state, dims=2 if is_recurrent else 1))[1]))
        if episode_steps > 1:
            advantages.append(estimate_episode_advantages(rewards[-episode_steps + 1:],
                                                          values[-episode_steps:],
                                                          discount, lam))
        values = np.array(values, dtype=np.float32)

        # write to the buffer
        advantages = np.hstack(advantages)
        buffer.fill(np.array(states, dtype=np.float32),
                    np.array(actions, dtype=np.float32 if is_continuous else np.int32),
                    np.array(action_probabilities, dtype=np.float32),
                    advantages,
                    advantages + values[:-1])
    else:
        buffer.pad_buffer()

    # normalize advantages
    buffer.normalize_advantages()

    if is_recurrent:
        # add batch dimension for optimization
        buffer.inject_batch_dimension()

    # convert buffer to dataset and save it to tf record
    dataset, stats = make_dataset_and_stats(buffer, is_shadow_brain=is_shadow_brain)
    dataset = dataset.map(tf_serialize_example)
    writer = tfl.data.experimental.TFRecordWriter(f"{STORAGE_DIR}/data_{pid}.tfrecord")
    writer.write(dataset)

    return stats


@ray.remote(num_cpus=1, num_gpus=0)
def evaluate(policy_tuple, env_name: str) -> Tuple[int, int]:
    """Evaluate one episode of the given environment following the given policy. Remote implementation."""
    environment = gym.make(env_name)

    if isinstance(policy_tuple, ModelTuple):
        model_builder = getattr(models, policy_tuple.model_builder)

        # recurrent policy needs batch size for statefulness
        policy, _, _ = model_builder(environment, **({"bs": 1} if "bs" in fargs(model_builder).args else {}))
        policy.set_weights(policy_tuple.weights)
    else:
        raise ValueError("Cannot handle given model type. Should be a ModelTuple.")

    if isinstance(environment.action_space, Discrete):
        continuous_control = False
    elif isinstance(environment.action_space, Box):
        continuous_control = True
    else:
        raise ValueError("Unknown action space.")

    is_recurrent = is_recurrent_model(policy)
    policy_act = act_discrete if not continuous_control else act_continuous

    done = False
    reward_trajectory = []
    length = 0
    state = parse_state(environment.reset())
    while not done:
        probabilities = policy(add_state_dims(state, dims=2 if is_recurrent else 1))
        action, action_prob = policy_act(probabilities)
        observation, reward, done, _ = environment.step(action)
        state = parse_state(observation)
        reward_trajectory.append(reward)
        length += 1

    return length, sum(reward_trajectory)


if __name__ == "__main__":
    os.chdir("../")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # env_n = "ShadowHand-v1"
    # env = gym.make(env_n)
    # p, v, j = build_shadow_brain(env, 1)
    # joint_tuple = ModelTuple(build_shadow_brain.__name__, j.get_weights())
    # if isinstance(env.observation_space, Dict) and "observation" in env.observation_space.sample():
    #     j(merge_into_batch([add_state_dims(env.observation_space.sample()["observation"], dims=1) for _ in range(1)]))

    env_n = "CartPole-v1"
    p, v, j = build_ffn_distinct_models(gym.make(env_n))
    joint_tuple = ModelTuple(build_ffn_distinct_models.__name__, j.get_weights())

    ray.init(local_mode=True)
    for i in tqdm(range(10000)):
        outs = [ray.get(collect.remote(joint_tuple, 1024, env_n, 0.99, 0.95, 16, 0)) for _ in range(2)]
