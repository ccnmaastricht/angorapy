#!/usr/bin/env python
"""Functions for gathering experience."""
import multiprocessing
from collections import namedtuple
from typing import Tuple, List

import numpy
import ray
import tensorflow as tf
from gym.spaces import Box

import models
from agent.core import estimate_advantage, normalize_advantages
from agent.policy import act_discrete, act_continuous
from environments import *

ExperienceBuffer = namedtuple("ExperienceBuffer", ["states", "actions", "action_probabilities", "returns", "advantages",
                                                   "episodes_completed", "episode_rewards", "episode_lengths"])
StatBundle = namedtuple("StatBundle", ["numb_completed_episodes", "numb_processed_frames",
                                       "episode_rewards", "episode_lengths"])
ModelTuple = namedtuple("ModelTuple", ["model_builder", "weights"])

RESERVED_GATHERING_CPUS = multiprocessing.cpu_count() - 2


@ray.remote(num_cpus=RESERVED_GATHERING_CPUS)
def collect(model, horizon: int, env_name: str, discount: float, lam: float):
    import tensorflow as tfl

    # build new environment for each collector to make multiprocessing possible
    env = gym.make(env_name)
    env_is_continuous = isinstance(env.action_space, Box)

    # load policy
    if isinstance(model, str):
        policy = tfl.keras.models.load_model(f"{model}/policy")
        critic = tfl.keras.models.load_model(f"{model}/value")
    elif isinstance(model, tuple):
        policy, _, _ = getattr(models, model[0].model_builder)(env)
        policy.set_weights(model[0].weights)
        _, critic, _ = getattr(models, model[1].model_builder)(env)
        critic.set_weights(model[1].weights)
    else:
        raise ValueError("Unknown input for model.")

    # trackers
    episodes_completed, current_episode_return, episode_steps = 0, 0, 1
    episode_rewards, episode_lengths = [], []

    # go for it
    states, rewards, actions, action_probabilities, t_is_terminal = [], [], [], [], []
    state = env.reset().astype(numpy.float32)
    act = act_continuous if env_is_continuous else act_discrete
    for t in range(horizon):
        # choose action and step
        action, action_probability = act(policy, numpy.expand_dims(state, axis=0))
        observation, reward, done, _ = env.step(numpy.atleast_1d(action) if env_is_continuous else action)

        # remember experience
        states.append(state)
        actions.append(action)
        action_probabilities.append(action_probability)
        rewards.append(reward)
        t_is_terminal.append(done == 1)
        current_episode_return += reward

        # next state
        if done:
            state = env.reset().astype(numpy.float32)
            episode_lengths.append(episode_steps)
            episode_rewards.append(current_episode_return)
            episodes_completed += 1
            episode_steps = 1
            current_episode_return = 0
        else:
            state = observation.astype(numpy.float32)
            episode_steps += 1

    value_predictions = critic(numpy.concatenate((states, [state]))).numpy().reshape([-1])
    advantages = estimate_advantage(rewards, value_predictions, t_is_terminal, gamma=discount, lam=lam)
    returns = numpy.add(advantages, value_predictions[:-1])
    advantages = normalize_advantages(advantages)

    # store this worker's gathering in a experience buffer
    buffer = ExperienceBuffer(states, actions, action_probabilities, returns, advantages,
                              episodes_completed, episode_rewards, episode_lengths)

    return buffer


def condense_worker_outputs(worker_outputs: List[ExperienceBuffer]) -> Tuple[tf.data.Dataset, StatBundle]:
    """Given a list of experience buffers produced by workers, produce a joined dataset and extract statistics."""
    completed_episodes = 0
    episode_rewards, episode_lengths = [], []
    joined_s_trajectory, joined_a_trajectory, joined_aprob_trajectory, joined_r, joined_advs = [], [], [], [], []

    # get and merge results
    for buffer in worker_outputs:
        joined_s_trajectory.extend(buffer.states)
        joined_a_trajectory.extend(buffer.actions)
        joined_aprob_trajectory.extend(buffer.action_probabilities)
        joined_r.extend(buffer.returns)
        joined_advs.extend(buffer.advantages)

        completed_episodes += buffer.episodes_completed
        episode_rewards.extend(buffer.episode_rewards)
        episode_lengths.extend(buffer.episode_lengths)

    numb_processed_frames = len(joined_s_trajectory)

    data = tf.data.Dataset.from_tensor_slices({
        "state": joined_s_trajectory,
        "action": joined_a_trajectory,
        "action_prob": joined_aprob_trajectory,
        "return": joined_r,
        "advantage": joined_advs
    })

    return data, StatBundle(completed_episodes, numb_processed_frames, episode_rewards, episode_lengths)
