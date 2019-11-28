#!/usr/bin/env python
"""Functions for gathering experience and communicating it to the main thread."""
import os
from inspect import getfullargspec as fargs
from typing import Tuple

import numpy as np
import ray
from gym.spaces import Box, Discrete

import models
from agent.core import estimate_episode_advantages
from agent.dataio import tf_serialize_example, make_dataset_and_stats
from agent.policy import act_discrete, act_continuous
from environments import *
from models import build_ffn_distinct_models
from utilities.const import STORAGE_DIR
from utilities.datatypes import ExperienceBuffer, ModelTuple
from utilities.util import parse_state, add_state_dims, is_recurrent_model


@ray.remote(num_cpus=1, num_gpus=0)
def collect(model, horizon: int, env_name: str, discount: float, lam: float, sub_sequence_length: int, pid: int):
    """Collect a batch shard of experience for a given number of timesteps."""

    # import here to avoid pickling errors
    import tensorflow as tfl

    # build new environment for each collector to make multiprocessing possible
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, Box)
    act = act_continuous if is_continuous else act_discrete

    # load policy TODO only one builder here
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
        assert horizon % sub_sequence_length == 0, "Subsequence length for TBPTT would require cutting of part of the" \
                                                   " observations."
    is_shadow_brain = "ShadowHand" in env_name

    # trackers
    episodes_completed, current_episode_return, episode_steps = 0, 0, 1
    episode_rewards, episode_lengths = [], []

    # go for it
    states, rewards, actions, action_probabilities, values, advantages = [], [], [], [], [], []
    state = parse_state(env.reset())
    for t in range(horizon):
        # based on the given state, predict action distribution and state value
        action_distribution, value = joint.predict(add_state_dims(state, dims=2 if is_recurrent else 1))
        states.append(state)
        values.append(np.squeeze(value).item())

        # from the action distribution sample an action and remember both the action and its probability
        action, action_probability = act(action_distribution)
        actions.append(action)
        action_probabilities.append(action_probability)

        # make a step based on the chosen action and collect the reward for this state
        observation, reward, done, _ = env.step(np.atleast_1d(action) if is_continuous else action)
        rewards.append(reward)
        current_episode_return += reward

        # depending on whether the state is terminal, choose the next state
        if done:
            state = parse_state(env.reset())

            # calculate advantages for the finished episode, where the last value is 0 since it refers to the terminal
            # state that we just observed
            advantages.append(
                estimate_episode_advantages(rewards[-episode_steps:], values[-episode_steps:] + [0], discount, lam)
            )

            # update/reset some statistics and trackers
            episode_lengths.append(episode_steps)
            episode_rewards.append(current_episode_return)
            episodes_completed += 1
            episode_steps = 1
            current_episode_return = 0
        else:
            state = parse_state(observation)
            episode_steps += 1
    env.close()

    values.append(joint(add_state_dims(state, dims=2 if is_recurrent else 1))[1].numpy().item())
    if episode_steps > 1:
        advantages.append(estimate_episode_advantages(rewards[-episode_steps + 1:],
                                                      values[-episode_steps:],
                                                      discount, lam))

    # convert lists to numpy arrays, excluding states as it can be multi-component
    actions = np.array(actions, dtype=np.float32 if is_continuous else np.int32)
    action_probabilities = np.array(action_probabilities, dtype=np.float32)
    advantages = np.hstack(advantages)
    values = np.array(values, dtype=np.float32)

    # get returns from advantages and values
    returns = advantages + values[:-1]

    # normalize advantages
    advantages = (advantages - advantages.mean()) / np.maximum(advantages.std(), 1e-6)

    # make TBPTT-compatible subsequences from transition vectors if model is recurrent
    if is_recurrent:
        num_sub_sequences = horizon // sub_sequence_length

        # states
        if is_shadow_brain:
            feature_tensors = [np.stack(list(map(lambda x: x[feature], states))) for feature in
                               range(len(states[0]))]
            states = tuple(map(lambda x: np.expand_dims(np.stack(np.split(x, num_sub_sequences)), axis=0),
                               feature_tensors))
        else:
            states = np.expand_dims(np.stack(np.split(states, num_sub_sequences, axis=0)), axis=0)

        # others, expanding dims to inject batch dimension
        actions = np.expand_dims(np.stack(np.split(actions, num_sub_sequences)), axis=0)
        action_probabilities = np.expand_dims(np.stack(np.split(action_probabilities, num_sub_sequences)), axis=0)
        advantages = np.expand_dims(np.stack(np.split(advantages, num_sub_sequences)), axis=0)
        returns = np.expand_dims(np.stack(np.split(returns, num_sub_sequences)), axis=0)

    # store this worker's gathering in a experience buffer
    buffer = ExperienceBuffer(states, actions, action_probabilities, returns,
                              advantages, episodes_completed, episode_rewards, episode_lengths)

    dataset, stats = make_dataset_and_stats(buffer)

    # save to tf record
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
    ray.get(collect.remote(joint_tuple, 256, env_n, 0.99, 0.95, 2, 0))
