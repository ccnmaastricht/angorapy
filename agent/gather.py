#!/usr/bin/env python
"""Functions for gathering experience and communicating it to the main thread."""
import itertools
import os
import random
from inspect import getfullargspec as fargs
from typing import Tuple, List

import numpy
import ray
import tensorflow as tf
from gym.spaces import Box
from tqdm import tqdm

import models
from agent.core import estimate_advantage
from agent.policy import act_discrete, act_continuous
from environments import *
from models import build_shadow_brain
from utilities.const import STORAGE_DIR
from utilities.datatypes import ExperienceBuffer, StatBundle, ModelTuple
from utilities.normalization import normalize_advantages
from utilities.util import parse_state, add_state_dims, is_recurrent_model


@ray.remote(num_cpus=1, num_gpus=0)
def collect(model, horizon: int, env_name: str, discount: float, lam: float, sub_sequence_length: int, pid: int):
    """Collect a batch shard of experience for a given number of timesteps."""

    # import here to avoid pickling errors
    import tensorflow as tfl

    # build new environment for each collector to make multiprocessing possible
    env = gym.make(env_name)
    env_is_continuous = isinstance(env.action_space, Box)
    act = act_continuous if env_is_continuous else act_discrete

    # load policy
    if isinstance(model, str):
        policy = tfl.keras.models.load_model(f"{model}/policy")
        critic = tfl.keras.models.load_model(f"{model}/value")
    elif isinstance(model, tuple):
        policy_builder = getattr(models, model[0].model_builder)
        value_builder = getattr(models, model[1].model_builder)

        # recurrent policy needs batch size for statefulness
        policy, _, _ = policy_builder(env, **({"bs": 1} if "bs" in fargs(policy_builder).args else {}))
        _, critic, _ = value_builder(env, **({"bs": 1} if "bs" in fargs(policy_builder).args else {}))

        # load the weights
        policy.set_weights(model[0].weights)
        critic.set_weights(model[1].weights)
    else:
        raise ValueError("Unknown input for model.")

    # check if there is a recurrent layer inside the model
    is_recurrent = is_recurrent_model(policy)
    if is_recurrent:
        assert horizon % sub_sequence_length == 0, "Subsequence length for TBPTT would require cutting of part of the" \
                                                   " observations."

    # trackers
    episodes_completed, current_episode_return, episode_steps = 0, 0, 1
    episode_rewards, episode_lengths = [], []

    # go for it
    states, rewards, actions, action_probabilities, t_is_terminal, values = [], [], [], [], [], []
    state = parse_state(env.reset())
    values.append(critic(add_state_dims(state, dims=2 if is_recurrent else 1)))
    for t in range(horizon):
        # choose action and step
        action, action_probability = act(policy, add_state_dims(state, dims=2 if is_recurrent else 1))
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
            state = parse_state(env.reset())
            episode_lengths.append(episode_steps)
            episode_rewards.append(current_episode_return)
            episodes_completed += 1
            episode_steps = 1
            current_episode_return = 0
        else:
            state = parse_state(observation)
            episode_steps += 1

        values.append(critic.predict(add_state_dims(state, dims=2 if is_recurrent else 1)))

    env.close()

    values = tfl.reshape(values, [-1])
    advantages = estimate_advantage(rewards, values, t_is_terminal, gamma=discount, lam=lam)
    returns = tfl.add(advantages, values[:-1])
    advantages = normalize_advantages(advantages)

    # make TBPTT-compatible subsequences from transition vectors if model is recurrent
    if is_recurrent:
        num_sub_sequences = horizon // sub_sequence_length

        # states
        feature_tensors = [tfl.stack(list(map(lambda x: x[feature], states))) for feature in range(len(states[0]))]
        states = list(zip(*list(map(lambda x: tfl.split(x, num_sub_sequences), feature_tensors))))

        # others
        actions = tfl.split(actions, num_sub_sequences)
        action_probabilities = tfl.split(action_probabilities, num_sub_sequences)
        advantages = tfl.split(advantages, num_sub_sequences)
        returns = tfl.split(returns, num_sub_sequences)

    # store this worker's gathering in a experience buffer
    buffer = ExperienceBuffer(states, actions, action_probabilities, returns, advantages,
                              episodes_completed, episode_rewards, episode_lengths)

    dataset, stats = make_dataset_and_stats(buffer)

    # save to tf record
    dataset = dataset.map(tf_serialize_example)
    writer = tfl.data.experimental.TFRecordWriter(f"{STORAGE_DIR}/data_{pid}.tfrecord")
    writer.write(dataset)

    return stats


def condense_stats(stat_bundles: List[StatBundle]) -> StatBundle:
    """Infer a single StatBundle from a list of StatBundles."""
    return StatBundle(
        numb_completed_episodes=sum([s.numb_completed_episodes for s in stat_bundles]),
        numb_processed_frames=sum([s.numb_processed_frames for s in stat_bundles]),
        episode_rewards=list(itertools.chain(*[s.episode_rewards for s in stat_bundles])),
        episode_lengths=list(itertools.chain(*[s.episode_lengths for s in stat_bundles]))
    )


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """"Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_flat_sample(s, a, ap, r, adv):
    """Serialize a sample from a dataset."""
    feature = {
        "state": _bytes_feature(tf.io.serialize_tensor(s)),
        "action": _bytes_feature(tf.io.serialize_tensor(a)),
        "action_prob": _bytes_feature(tf.io.serialize_tensor(ap)),
        "return": _bytes_feature(tf.io.serialize_tensor(r)),
        "advantage": _bytes_feature(tf.io.serialize_tensor(adv))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_shadow_hand_sample(sv, sp, st, sg, a, ap, r, adv):
    """Serialize a multi-input (shadow hand) sample from a dataset."""
    feature = {
        "in_vision": _bytes_feature(tf.io.serialize_tensor(sv)),
        "in_proprio": _bytes_feature(tf.io.serialize_tensor(sp)),
        "in_touch": _bytes_feature(tf.io.serialize_tensor(st)),
        "in_goal": _bytes_feature(tf.io.serialize_tensor(sg)),
        "action": _bytes_feature(tf.io.serialize_tensor(a)),
        "action_prob": _bytes_feature(tf.io.serialize_tensor(ap)),
        "return": _bytes_feature(tf.io.serialize_tensor(r)),
        "advantage": _bytes_feature(tf.io.serialize_tensor(adv))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(sample):
    """TF wrapper for serialization function."""
    if "state" in sample:
        inputs = (sample["state"],)
        serializer = serialize_flat_sample
    else:
        inputs = (sample["in_vision"], sample["in_proprio"], sample["in_touch"], sample["in_goal"])
        serializer = serialize_shadow_hand_sample
    inputs += (sample["action"], sample["action_prob"], sample["return"], sample["advantage"])

    tf_string = tf.py_function(serializer, inputs, tf.string)
    return tf.reshape(tf_string, ())


def make_dataset_and_stats(buffer: ExperienceBuffer):
    """Make dataset object and StatBundle from ExperienceBuffer."""
    completed_episodes = buffer.episodes_completed
    numb_processed_frames = len(buffer.states)

    if isinstance(buffer.states[0], numpy.ndarray):
        dataset = tf.data.Dataset.from_tensor_slices({
            "state": buffer.states,
            "action": buffer.actions,
            "action_prob": buffer.action_probabilities,
            "return": buffer.returns,
            "advantage": buffer.advantages
        })
    elif isinstance(buffer.states[0], Tuple):
        dataset = tf.data.Dataset.from_tensor_slices({
            "in_vision": [x[0] for x in buffer.states],
            "in_proprio": [x[1] for x in buffer.states],
            "in_touch": [x[2] for x in buffer.states],
            "in_goal": [x[3] for x in buffer.states],
            "action": buffer.actions,
            "action_prob": buffer.action_probabilities,
            "return": buffer.returns,
            "advantage": buffer.advantages
        })
    else:
        raise NotImplementedError(f"Cannot handle state type {type(buffer.states[0])}")

    stats = StatBundle(
        completed_episodes,
        numb_processed_frames,
        buffer.episode_rewards,
        buffer.episode_lengths
    )

    return dataset, stats


def read_dataset_from_storage(dtype_actions: tf.dtypes.DType, is_shadow_hand: bool, shuffle: bool = True):
    """Read all files in storage into a tf record dataset without actually loading everything into memory."""
    feature_description = {
        "action": tf.io.FixedLenFeature([], tf.string),
        "action_prob": tf.io.FixedLenFeature([], tf.string),
        "return": tf.io.FixedLenFeature([], tf.string),
        "advantage": tf.io.FixedLenFeature([], tf.string)
    }

    # add states
    if not is_shadow_hand:
        feature_description["state"] = tf.io.FixedLenFeature([], tf.string)
    else:
        feature_description.update({
            "in_vision": tf.io.FixedLenFeature([], tf.string),
            "in_proprio": tf.io.FixedLenFeature([], tf.string),
            "in_touch": tf.io.FixedLenFeature([], tf.string),
            "in_goal": tf.io.FixedLenFeature([], tf.string),
        })

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        if not is_shadow_hand:
            parsed["state"] = tf.io.parse_tensor(parsed["state"], out_type=tf.float32)
        else:
            parsed["in_vision"] = tf.io.parse_tensor(parsed["in_vision"], out_type=tf.float32)
            parsed["in_proprio"] = tf.io.parse_tensor(parsed["in_proprio"], out_type=tf.float32)
            parsed["in_touch"] = tf.io.parse_tensor(parsed["in_touch"], out_type=tf.float32)
            parsed["in_goal"] = tf.io.parse_tensor(parsed["in_goal"], out_type=tf.float32)
        parsed["action"] = tf.io.parse_tensor(parsed["action"], out_type=dtype_actions)
        parsed["action_prob"] = tf.io.parse_tensor(parsed["action_prob"], out_type=tf.float32)
        parsed["return"] = tf.io.parse_tensor(parsed["return"], out_type=tf.float32)
        parsed["advantage"] = tf.io.parse_tensor(parsed["advantage"], out_type=tf.float32)

        return parsed

    files = [os.path.join(STORAGE_DIR, name) for name in os.listdir(STORAGE_DIR)]
    if shuffle:
        random.shuffle(files)
    serialized_dataset = tf.data.TFRecordDataset(files)
    serialized_dataset = serialized_dataset.map(_parse_function)

    return serialized_dataset


@DeprecationWarning
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


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env_name = "ShadowHand-v1"
    policy, value, joint = build_shadow_brain(gym.make(env_name), 1)

    policy_tuple = ModelTuple(build_shadow_brain.__name__, policy.get_weights())
    critic_tuple = ModelTuple(build_shadow_brain.__name__, value.get_weights())
    mods = (policy_tuple, critic_tuple)

    ray.init(local_mode=True)
    collect.remote(mods, 1024, env_name, 0.99, 0.95, 0)
