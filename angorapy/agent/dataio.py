#!/usr/bin/env python
"""Data reading and writing utilities for distributed learning."""
import os
import random
import re
from functools import partial
from typing import Union, Tuple, List

import numpy as np
import tensorflow as tf
from mpi4py import MPI

from angorapy.common.const import STORAGE_DIR
from angorapy.common.senses import Sensation
from angorapy.utilities.datatypes import StatBundle
from angorapy.common.data_buffers import ExperienceBuffer, TimeSequenceExperienceBuffer


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """"Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample(*args, feature_names):
    """Serialize a multi-input (shadow hand) sample from a dataset."""
    assert len(args) == len(feature_names), "Sample serialization reveived an unequal number of features than names."

    feature = {}
    for i, feature_name in enumerate(feature_names):
        feature[feature_name] = _bytes_feature(tf.io.serialize_tensor(args[i]))

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(sample):
    """TF wrapper for serialization function."""
    feature_names = ([sense for sense in ["vision", "somatosensation", "proprioception", "goal"] if sense in sample]
                     + ["action", "action_prob", "return", "advantage", "value", "done", "mask"])

    inputs = [sample[f] for f in feature_names]

    tf_string = tf.py_function(partial(serialize_sample, feature_names=feature_names), inputs, tf.string)
    return tf.reshape(tf_string, ())


def make_dataset_and_stats(buffer: ExperienceBuffer) -> Tuple[tf.data.Dataset, StatBundle]:
    """Make dataset object and StatBundle from ExperienceBuffer."""
    completed_episodes = buffer.episodes_completed
    numb_processed_frames = buffer.capacity * buffer.seq_length

    # build the dataset
    tensor_slices = {
        "action": buffer.actions,
        "action_prob": buffer.action_probabilities,
        "return": buffer.returns,
        "advantage": buffer.advantages,
        "value": buffer.values,
        "done": buffer.dones,
        "mask": buffer.mask
    }

    tensor_slices.update(buffer.states)

    dataset = tf.data.TFRecordDataset.from_tensor_slices(tensor_slices)

    # make statistics object
    underflow = None
    if isinstance(buffer, TimeSequenceExperienceBuffer):
        underflow = round(1 - (buffer.true_number_of_transitions / (buffer.capacity * buffer.seq_length)), 2)

    stats = StatBundle(
        completed_episodes,
        numb_processed_frames,
        buffer.episode_rewards,
        buffer.episode_lengths,
        tbptt_underflow=underflow,
        per_receptor_mean={
            # mean over all buf the last dimension to mean per receptor (reducing batch and time dimensions)
            # todo ignore empty masked states
            sense: np.mean(buffer.states[sense], axis=tuple(range(len(buffer.states[sense].shape)))[:-1])
            for sense in buffer.states.keys()
        },
        auxiliary_performances=buffer.auxiliary_performances
    )

    return dataset, stats


def read_dataset_from_storage(dtype_actions: tf.dtypes.DType, id_prefix: Union[str, int], responsive_senses: List[str],
                              shuffle: bool = True, worker_ids: List = None):
    """Read all files in storage into a tf record dataset without actually loading everything into memory.

    Args:
        dtype_actions:      datatype of the actions
        id_prefix:          prefix (usually agent id) by which to filter available experience files
        responsive_senses:  list of senses (string represented, in ["proprioception", "vision", "somatosensation",
                            "goal"]) that the agent utilizes
        shuffle:            whether or not to shuffle the datafiles
    """
    assert all(r in Sensation.sense_names for r in responsive_senses)

    feature_description = {
        "action": tf.io.FixedLenFeature([], tf.string),
        "action_prob": tf.io.FixedLenFeature([], tf.string),
        "return": tf.io.FixedLenFeature([], tf.string),
        "advantage": tf.io.FixedLenFeature([], tf.string),
        "value": tf.io.FixedLenFeature([], tf.string),
        "done": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),

        # STATES
        **{sense: tf.io.FixedLenFeature([], tf.string) for sense in responsive_senses}
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)

        parsed["action"] = tf.io.parse_tensor(parsed["action"], out_type=dtype_actions)
        parsed["action_prob"] = tf.io.parse_tensor(parsed["action_prob"], out_type=tf.float32)
        parsed["return"] = tf.io.parse_tensor(parsed["return"], out_type=tf.float32)
        parsed["advantage"] = tf.io.parse_tensor(parsed["advantage"], out_type=tf.float32)
        parsed["value"] = tf.io.parse_tensor(parsed["value"], out_type=tf.float32)
        parsed["done"] = tf.io.parse_tensor(parsed["done"], out_type=tf.bool)
        parsed["mask"] = tf.io.parse_tensor(parsed["mask"], out_type=tf.bool)

        for sense in responsive_senses:
            parsed[sense] = tf.io.parse_tensor(parsed[sense], out_type=tf.float32)

        return parsed

    worker_id_regex = "[0-9]*"
    if worker_ids is not None:
        worker_id_regex = "(" + "|".join(map(str, worker_ids)) + ")"

    files = [os.path.join(STORAGE_DIR, name) for name in os.listdir(STORAGE_DIR) if
             re.match(f"{id_prefix}_data_{worker_id_regex}\.tfrecord", name)]

    random.shuffle(files) if shuffle else None

    serialized_dataset = tf.data.TFRecordDataset(files)
    serialized_dataset = serialized_dataset.map(_parse_function)

    return serialized_dataset
