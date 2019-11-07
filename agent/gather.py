#!/usr/bin/env python
"""Functions for gathering experience."""
import multiprocessing
import os
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
STORAGE_DIR = "storage/experience/"


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


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """"Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample(s, a, ap, r, adv):
    """Serialize a sample from a dataset."""
    feature = {
        "state": _bytes_feature(tf.io.serialize_tensor(s)),
        "action": _bytes_feature(tf.io.serialize_tensor(a)),
        "action_prob": _float_feature(ap),
        "return": _float_feature(r),
        "advantage": _float_feature(adv)
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(sample):
    tf_string = tf.py_function(serialize_sample, (sample["state"],
                                                  sample["action"],
                                                  sample["action_prob"],
                                                  sample["return"],
                                                  sample["advantage"],), tf.string)
    return tf.reshape(tf_string, ())


@ray.remote(num_cpus=RESERVED_GATHERING_CPUS)
def collect(model, horizon: int, env_name: str, discount: float, lam: float, pid: int):
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

    # save to tf record
    dataset, stats = condense_worker_outputs([buffer])

    dataset = dataset.map(tf_serialize_example)
    writer = tfl.data.experimental.TFRecordWriter(f"storage/experience/data_{pid}.tfrecord")
    writer.write(dataset)

    return stats


def read_dataset_from_storage(dtype_actions: tf.dtypes.DType):
    """Read all files in storage into a tf record dataset without actually loading everything into memory."""
    feature_description = {
        "state": tf.io.FixedLenFeature([], tf.string),
        "action": tf.io.FixedLenFeature([], tf.string),
        "action_prob": tf.io.FixedLenFeature([], tf.float32),
        "return": tf.io.FixedLenFeature([], tf.float32),
        "advantage": tf.io.FixedLenFeature([], tf.float32)
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        parsed["state"] = tf.io.parse_tensor(parsed["state"], out_type=tf.float32)
        parsed["action"] = tf.io.parse_tensor(parsed["action"], out_type=dtype_actions)
        return parsed

    serialized_dataset = tf.data.TFRecordDataset([os.path.join(STORAGE_DIR, name) for name in os.listdir(STORAGE_DIR)])
    serialized_dataset = serialized_dataset.map(_parse_function)

    return serialized_dataset