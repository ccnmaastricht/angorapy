#!/usr/bin/env python
"""Gatherer classes."""
from collections import namedtuple
from multiprocessing import Queue
from typing import Tuple

import gym
import numpy
import ray
import tensorflow as tf
from gym.spaces import Box

from agent.core import estimate_advantage, normalize_advantages

ExperienceBuffer = namedtuple("ExperienceBuffer", ["states", "actions", "action_probabilities", "returns", "advantages",
                                                   "episodes_completed", "episode_rewards", "episode_lengths"])
StatBundle = namedtuple("StatBundle", ["numb_completed_episodes", "numb_processed_frames",
                                       "episode_rewards", "episode_lengths"])


@ray.remote
def collect(agent, horizon: int, env_name: str):
    # build new environment for each collector to make multiprocessing possible
    env = gym.make(env_name)
    env_is_continuous = isinstance(env.action_space, Box)

    # trackers
    episodes_completed = 0
    current_episode_return = 0
    episode_rewards = []
    episode_lengths = []
    episode_steps = 1

    # go for it
    states, rewards, actions, action_probabilities, t_is_terminal = [], [], [], [], []
    state = env.reset()
    for t in range(horizon):
        # choose action and step
        action, action_probability = agent.act(numpy.reshape(state, [1, -1]))
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
            state = env.reset()
            episode_lengths.append(episode_steps)
            episode_rewards.append(current_episode_return)
            episodes_completed += 1
            episode_steps = 1
            current_episode_return = 0
        else:
            state = observation
            episode_steps += 1

    value_predictions = [agent.critic(tf.reshape(s, [1, -1]))[0][0].numpy() for s in states + [state]]
    advantages = estimate_advantage(rewards, value_predictions, t_is_terminal, gamma=agent.discount, lam=agent.lam)
    returns = numpy.add(advantages, value_predictions[:-1])
    advantages = normalize_advantages(advantages)

    # store this worker's gathering in the shared memory
    buffer = ExperienceBuffer(states, actions, action_probabilities, returns, advantages,
                              episodes_completed, episode_rewards, episode_lengths)
    # queue.put(buffer, False)

    return buffer


def condense_buffer_queue(queue_of_buffers) -> Tuple[tf.data.Dataset, StatBundle]:
    # get and merge results
    episode_rewards = []
    episode_lengths = []
    completed_episodes = 0
    joined_s_trajectory, joined_a_trajectory, joined_aprob_trajectory, joined_r, joined_advs = [], [], [], [], []
    # while not queue_of_buffers.empty():
    for buffer in queue_of_buffers:
        # buffer = queue_of_buffers.get()
        joined_s_trajectory.extend(buffer.states)
        joined_a_trajectory.extend(buffer.actions)
        joined_aprob_trajectory.extend(buffer.action_probabilities)
        joined_r.extend(buffer.returns)
        joined_advs.extend(buffer.advantages)

        completed_episodes += buffer.episodes_completed
        episode_rewards.extend(buffer.episode_rewards)
        episode_lengths.extend(buffer.episode_lengths)

    numb_processed_frames = len(joined_s_trajectory)

    return tf.data.Dataset.from_tensor_slices({
        "state": joined_s_trajectory,
        "action": joined_a_trajectory,
        "action_prob": joined_aprob_trajectory,
        "return": joined_r,
        "advantage": joined_advs
    }), StatBundle(completed_episodes, numb_processed_frames, episode_rewards, episode_lengths)


if __name__ == "__main__":
    import pickle
    with open("test.obj", "wb") as f:
        pickle.dump(ExperienceBuffer(10, 10, 10, 10, 10, 10, 10, 10), f)
