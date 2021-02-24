#!/usr/bin/env python
"""Data reading and writing utilities for distributed learning."""
import os
import random
import re
from functools import partial
from typing import Union, Tuple, List

import numpy as np
import tensorflow as tf
from common.senses import Sensation

from utilities.const import STORAGE_DIR
from utilities.datatypes import ExperienceBuffer, StatBundle, TimeSequenceExperienceBuffer


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """"Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_flat_sample(s, a, ap, r, adv, v):
    """Serialize a sample from a dataset."""
    feature = {
        "state": _bytes_feature(tf.io.serialize_tensor(s)),
        "step_tuple": _bytes_feature(tf.io.serialize_tensor(a)),
        "action_prob": _bytes_feature(tf.io.serialize_tensor(ap)),
        "return": _bytes_feature(tf.io.serialize_tensor(r)),
        "advantage": _bytes_feature(tf.io.serialize_tensor(adv)),
        "value": _bytes_feature(tf.io.serialize_tensor(v))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_sampleold(sv, sp, st, sg, a, ap, r, adv, v, feature_names):
    """Serialize a multi-input (shadow hand) sample from a dataset."""
    feature = {
        "vision": _bytes_feature(tf.io.serialize_tensor(sv)),
        "somatosensation": _bytes_feature(tf.io.serialize_tensor(st)),
        "proprioception": _bytes_feature(tf.io.serialize_tensor(sp)),
        "goal": _bytes_feature(tf.io.serialize_tensor(sg)),

        "step_tuple": _bytes_feature(tf.io.serialize_tensor(a)),
        "action_prob": _bytes_feature(tf.io.serialize_tensor(ap)),
        "return": _bytes_feature(tf.io.serialize_tensor(r)),
        "advantage": _bytes_feature(tf.io.serialize_tensor(adv)),
        "value": _bytes_feature(tf.io.serialize_tensor(v))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


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
                     + ["step_tuple", "action_prob", "return", "advantage", "value"])

    inputs = [sample[f] for f in feature_names]

    tf_string = tf.py_function(partial(serialize_sample, feature_names=feature_names), inputs, tf.string)
    return tf.reshape(tf_string, ())


def make_dataset_and_stats(buffer: ExperienceBuffer) -> Tuple[tf.data.Dataset, StatBundle]:
    """Make dataset object and StatBundle from ExperienceBuffer."""
    completed_episodes = buffer.episodes_completed
    numb_processed_frames = buffer.capacity

    # build the dataset
    tensor_slices = {
        "step_tuple": buffer.actions,
        "action_prob": buffer.action_probabilities,
        "return": buffer.returns,
        "advantage": buffer.advantages,
        "value": buffer.advantages,
    }

    for sense in buffer.states[0].dict().keys():
        tensor_slices[sense] = [e[sense] for e in buffer.states]

    dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)

    # make statistics object
    underflow = None
    if isinstance(buffer, TimeSequenceExperienceBuffer):
        underflow = round(1 - buffer.true_number_of_transitions / buffer.capacity, 2)

    stats = StatBundle(
        completed_episodes,
        numb_processed_frames,
        buffer.episode_rewards,
        buffer.episode_lengths,
        tbptt_underflow=underflow
    )

    return dataset, stats


def read_dataset_from_storage(dtype_actions: tf.dtypes.DType, id_prefix: Union[str, int], responsive_senses: List[str],
                              shuffle: bool = True):
    """Read all files in storage into a tf record dataset without actually loading everything into memory.

    Args:
        dtype_actions:      datatype of the actions
        id_prefix:          prefix (usually agent id) by which to filter available experience files
        responsive_senses:  list of senses (string represented, in ["proprioception", "vision", "somatosensation",
                            "goal"]) that the agent utilizes
        shuffle:            whether or not to shuffle the datafiles
    """
    assert all(r in ["proprioception", "vision", "somatosensation", "goal"] for r in responsive_senses)

    feature_description = {
        "step_tuple": tf.io.FixedLenFeature([], tf.string),
        "action_prob": tf.io.FixedLenFeature([], tf.string),
        "return": tf.io.FixedLenFeature([], tf.string),
        "advantage": tf.io.FixedLenFeature([], tf.string),
        "value": tf.io.FixedLenFeature([], tf.string),

        # STATES
        **{sense: tf.io.FixedLenFeature([], tf.string) for sense in responsive_senses}
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)

        parsed["step_tuple"] = tf.io.parse_tensor(parsed["step_tuple"], out_type=dtype_actions)
        parsed["action_prob"] = tf.io.parse_tensor(parsed["action_prob"], out_type=tf.float32)
        parsed["return"] = tf.io.parse_tensor(parsed["return"], out_type=tf.float32)
        parsed["advantage"] = tf.io.parse_tensor(parsed["advantage"], out_type=tf.float32)
        parsed["value"] = tf.io.parse_tensor(parsed["value"], out_type=tf.float32)

        for sense in responsive_senses:
            parsed[sense] = tf.io.parse_tensor(parsed[sense], out_type=tf.float32)

        return parsed

    files = [os.path.join(STORAGE_DIR, name) for name in os.listdir(STORAGE_DIR) if
             re.match(f"{id_prefix}_.[0-9]*", name)]
    random.shuffle(files) if shuffle else None

    serialized_dataset = tf.data.TFRecordDataset(files)
    serialized_dataset = serialized_dataset.map(_parse_function)

    return serialized_dataset


if __name__ == '__main__':
    s = Sensation(np.array([1, 3, 4]), np.array([1, 3, 3]), np.array([1, 3, 2]))

    print()
    print(s)
    s.inject_leading_dims(time=True)
    print(s)
