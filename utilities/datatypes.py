#!/usr/bin/env python
"""Custom datatypes form simplifying data communication."""
from collections import namedtuple
from typing import List, Union

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from numpy import ndarray as arr

StatBundle = namedtuple("StatBundle", ["numb_completed_episodes", "numb_processed_frames",
                                       "episode_rewards", "episode_lengths"])
ModelTuple = namedtuple("ModelTuple", ["model_builder", "weights"])


class ExperienceBuffer:
    """Buffer for experience gathered in an environment."""

    def __init__(self, states: Union[List, arr], actions: Union[List, arr], action_probabilities: Union[List, arr],
                 returns: Union[List, arr], advantages: Union[List, arr], episodes_completed: int,
                 episode_rewards: List[int], episode_lengths: List[int]):

        self.buffer_size = 0
        self.episode_lengths = episode_lengths
        self.episode_rewards = episode_rewards
        self.episodes_completed = episodes_completed
        self.advantages = advantages
        self.returns = returns
        self.action_probabilities = action_probabilities
        self.actions = actions
        self.states = states

    def fill(self, s, a, ap, ret, adv):
        """Fill the buffer with 5-tuple of experience."""
        assert np.all([len(s), len(a), len(ap), len(ret), len(adv)] == len(s)), "Inconsistent input sizes."

        self.buffer_size = len(adv)
        self.advantages, self.returns, self.action_probabilities, self.actions, self.states = adv, ret, ap, a, s

    def push_seq_to_buffer(self, states: List[arr], actions: List[arr], action_probabilities: List[arr],
                           advantages: List[arr], is_multi_feature: bool, is_continuous: bool):
        """Push a sequence to the buffer, constructed from given lists of values."""
        if is_multi_feature:
            self.states.append([np.stack(list(map(lambda s: s[f_id], states))) for f_id in range(len(states[0]))])
        else:
            self.states.append(np.array(states, dtype=np.float32))

        self.actions.append(np.array(actions, dtype=np.float32 if is_continuous else np.int32))
        self.action_probabilities.append(np.array(action_probabilities, dtype=np.float32))
        self.advantages.append(advantages)  # TODO correct last value

    def pad_buffer(self):
        """Pad the buffer with zeros to an equal sequence length."""
        assert np.all([isinstance(f, list) for f in [self.states, self.actions, self.action_probabilities,
                                                     self.returns, self.advantages]])

        self.states = pad_sequences(self.states, padding="post", dtype=tf.float32)
        self.action_probabilities = pad_sequences(self.action_probabilities, padding="post", dtype=tf.float32)
        self.actions = pad_sequences(self.actions, padding="post", dtype=tf.float32)
        self.returns = pad_sequences(self.returns, padding="post", dtype=tf.float32)
        self.advantages = pad_sequences(self.advantages, padding="post", dtype=tf.float32)

    def normalize_advantages(self):
        """Normalize the buffered advantages using z-scores. This requires the sequences to be of equal lengths,
        hence if this is not guaranteed during pushing to the buffer, pad_buffer has to be called first."""
        assert isinstance(self.advantages, arr), "Advantages are still a list of sequences. " \
                                                 "You should call pad_buffer first."
        rank = len(self.advantages.shape)
        assert rank in [1, 2], f"Illegal rank of advantage tensor, should be 1 or 2 but is {rank}."
        is_recurrent = rank == 2

        mean = np.mean(self.advantages)
        std = np.maximum(np.std(self.advantages), 1e-6)

        if is_recurrent:
            self.advantages = list(map(lambda adv: (adv - mean) / std, self.advantages))
        else:
            self.advantages = (self.advantages - mean) / std

    @staticmethod
    def new_empty():
        """Return an empty buffer."""
        return ExperienceBuffer(states=[], actions=[], action_probabilities=[], returns=[], advantages=[],
                                episodes_completed=0, episode_rewards=[], episode_lengths=[])
