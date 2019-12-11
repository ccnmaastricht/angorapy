#!/usr/bin/env python
"""Custom data types for simplifying data communication."""
import itertools
import logging
from collections import namedtuple
from typing import List, Union

import gym
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from numpy import ndarray as arr

from utilities.util import add_state_dims, env_extract_dims

StatBundle = namedtuple("StatBundle", ["numb_completed_episodes", "numb_processed_frames",
                                       "episode_rewards", "episode_lengths"])
ModelTuple = namedtuple("ModelTuple", ["model_builder", "weights"])


class ExperienceBuffer:
    """Buffer for experience gathered in an environment."""

    def __init__(self, states: Union[List, arr], actions: Union[List, arr], action_probabilities: Union[List, arr],
                 returns: Union[List, arr], advantages: Union[List, arr], episodes_completed: int,
                 episode_rewards: List[int], episode_lengths: List[int], is_multi_feature: bool, is_continuous: bool):

        self.is_continuous = is_continuous
        self.is_multi_feature = is_multi_feature

        self.size = actions.shape[0] if actions != [] else 0
        self.filled = 0

        self.episode_lengths = episode_lengths
        self.episode_rewards = episode_rewards
        self.episodes_completed = episodes_completed
        self.advantages = advantages
        self.returns = returns
        self.action_probabilities = action_probabilities
        self.actions = actions
        self.states = states

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.filled}/{self.size}]"

    def fill(self, s, a, ap, ret, adv):
        """Fill the buffer with 5-tuple of experience."""
        assert np.all(np.array([len(s), len(a), len(ap), len(ret), len(adv)]) == len(s)), "Inconsistent input sizes."

        self.size = len(adv)
        self.advantages, self.returns, self.action_probabilities, self.actions, self.states = adv, ret, ap, a, s

    def normalize_advantages(self):
        """Normalize the buffered advantages using z-scores. This requires the sequences to be of equal lengths,
        hence if this is not guaranteed during pushing to the buffer, pad_buffer has to be called first."""
        rank = len(self.advantages.shape)
        assert rank in [1, 2], f"Illegal rank of advantage tensor, should be 1 or 2 but is {rank}."

        mean = np.mean(self.advantages)
        std = np.maximum(np.std(self.advantages), 1e-6)
        self.advantages = (self.advantages - mean) / std

    def inject_batch_dimension(self):
        """Add a batch dimension to the buffered experience."""
        self.states = add_state_dims(self.states, axis=0)
        self.actions = np.expand_dims(self.actions, axis=0)
        self.action_probabilities = np.expand_dims(self.action_probabilities, axis=0)
        self.returns = np.expand_dims(self.returns, axis=0)
        self.advantages = np.expand_dims(self.advantages, axis=0)

    @staticmethod
    def new_empty(is_continuous, is_multi_feature):
        """Return an empty buffer."""
        return ExperienceBuffer(states=[],
                                actions=[],
                                action_probabilities=[],
                                returns=[],
                                advantages=[],
                                episodes_completed=0, episode_rewards=[], episode_lengths=[],
                                is_continuous=is_continuous,
                                is_multi_feature=is_multi_feature)

    @staticmethod
    def new(env: gym.Env, size: int, is_continuous, is_multi_feature):
        """Return an empty buffer."""
        state_dim, action_dim = env_extract_dims(env)

        if isinstance(state_dim, int):
            state_buffer = np.zeros((size, action_dim))
        else:
            state_buffer = tuple(np.zeros((size,) + shape) for shape in state_dim)
        return ExperienceBuffer(states=state_buffer,
                                actions=np.zeros((size, action_dim)),
                                action_probabilities=np.zeros((size,)),
                                returns=np.zeros((size,)),
                                advantages=np.zeros((size,)),
                                episodes_completed=0, episode_rewards=[], episode_lengths=[],
                                is_continuous=is_continuous,
                                is_multi_feature=is_multi_feature)


class TimeSequenceExperienceBuffer(ExperienceBuffer):
    """Experience Buffer for TimeSequence Data"""

    def __init__(self, states: Union[List, arr], actions: Union[List, arr], action_probabilities: Union[List, arr],
                 returns: Union[List, arr], advantages: Union[List, arr], episodes_completed: int,
                 episode_rewards: List[int], episode_lengths: List[int], is_multi_feature: bool, is_continuous: bool):

        super().__init__(states, actions, action_probabilities, returns, advantages, episodes_completed,
                         episode_rewards, episode_lengths, is_multi_feature, is_continuous)

        self.true_number_of_transitions = 0
        self.advantage_mask = np.ones(advantages.shape)

    def push_seq_to_buffer(self, states: List[arr], actions: List[arr], action_probabilities: List[arr],
                           advantages: List[arr], returns: List[arr]):
        """Push a sequence to the buffer, constructed from given lists of values."""
        seq_length = len(actions)
        self.true_number_of_transitions += seq_length

        if self.is_multi_feature:
            # can I point out for a second that numpy slicing is beautiful as fu**
            states = [np.stack(list(map(lambda s: s[f_id], states))) for f_id in range(len(states[0]))]
        else:
            states = np.stack(states)
        advantages = np.stack(advantages)

        self.states[self.filled, :seq_length, ...] = states
        self.actions[self.filled, :seq_length, ...] = np.stack(actions)
        self.action_probabilities[self.filled, :seq_length] = action_probabilities
        self.advantages[self.filled, :seq_length] = advantages
        self.advantage_mask[self.filled, :seq_length] = np.zeros(advantages.shape)
        self.returns[self.filled, :seq_length] = returns

        self.filled += 1

    def normalize_advantages(self):
        """Normalize the buffered advantages using z-scores. This requires the sequences to be of equal lengths,
        hence if this is not guaranteed during pushing to the buffer, pad_buffer has to be called first."""
        rank = len(self.advantages.shape)
        assert rank in [1, 2], f"Illegal rank of advantage tensor, should be 1 or 2 but is {rank}."

        # TODO could better keep track of already masked array in self.advantages?
        masked_advantages = np.ma.masked_array(self.advantages, self.advantage_mask)
        mean = masked_advantages.mean()
        std = np.maximum(masked_advantages.std(), 1e-6)
        self.advantages = (self.advantages - mean) / std

    @DeprecationWarning
    def pad_buffer(self):
        """Pad the buffer with zeros to an equal sequence length."""
        assert np.all([isinstance(f, list) for f in [self.states, self.actions, self.action_probabilities,
                                                     self.returns,
                                                     self.advantages]]), "Some part of the experience " \
                                                                         "is not a list but you want to pad."

        assert np.all([len(f) > 0 for f in [self.states, self.actions, self.action_probabilities,
                                            self.returns,
                                            self.advantages]]), "There cannot be an empty s/a/ap/r/adv when padding."

        size_diffs = np.array([len(self.actions), len(self.action_probabilities), len(self.returns),
                               len(self.advantages)]) == len(self.states)
        if not np.all(size_diffs):
            logging.warning(f"Buffer contains inconsistent lengths of s/a/ap/r/adv prior to padding [{size_diffs}].")

        self.states = pad_sequences(self.states, padding="post", dtype=np.float32)
        self.action_probabilities = pad_sequences(self.action_probabilities, padding="post", dtype=np.float32)
        self.actions = pad_sequences(self.actions, padding="post")
        self.returns = pad_sequences(self.returns, padding="post", dtype=np.float32)
        self.advantages = pad_sequences(self.advantages, padding="post", dtype=np.float32)

    @staticmethod
    def new(env: gym.Env, size: int, seq_len: int, is_continuous, is_multi_feature):
        """Return an empty buffer for sequences."""
        state_dim, action_dim = env_extract_dims(env)

        if isinstance(state_dim, int):
            state_buffer = np.zeros((size, seq_len, state_dim), dtype=np.float32)
        else:
            state_buffer = tuple(np.zeros((size, seq_len) + shape) for shape in state_dim)
        return TimeSequenceExperienceBuffer(states=state_buffer,
                                            actions=np.zeros((size, seq_len) + ((action_dim, ) if is_continuous else ()),
                                                             dtype=np.float32 if is_continuous else np.int32),
                                            action_probabilities=np.zeros((size, seq_len,), dtype=np.float32),
                                            returns=np.zeros((size, seq_len), dtype=np.float32),
                                            advantages=np.zeros((size, seq_len), dtype=np.float32),
                                            episodes_completed=0, episode_rewards=[], episode_lengths=[],
                                            is_continuous=is_continuous, is_multi_feature=is_multi_feature)


def condense_stats(stat_bundles: List[StatBundle]) -> StatBundle:
    """Infer a single StatBundle from a list of StatBundles."""
    return StatBundle(
        numb_completed_episodes=sum([s.numb_completed_episodes for s in stat_bundles]),
        numb_processed_frames=sum([s.numb_processed_frames for s in stat_bundles]),
        episode_rewards=list(itertools.chain(*[s.episode_rewards for s in stat_bundles])),
        episode_lengths=list(itertools.chain(*[s.episode_lengths for s in stat_bundles]))
    )


if __name__ == '__main__':
    from environments import *

    environment = gym.make("ShadowHand-v1")
    buffer = TimeSequenceExperienceBuffer.new(environment, 10, 16, True, True)
    print(buffer)
