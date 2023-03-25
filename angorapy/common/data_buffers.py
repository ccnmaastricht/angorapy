from typing import List, Tuple, Dict

import numpy as np
from numpy.core._multiarray_umath import ndarray as arr

from angorapy.common.senses import Sensation, stack_sensations


class ExperienceBuffer:
    """Buffer for experience gathered by a single worker."""

    def __init__(self, capacity: int, state_dim: Dict[str, Tuple[int]], action_dim: Tuple[int], is_continuous: bool):
        """Initialize the buffer with empty numpy arrays.

        Individual buffers include a leading batch dimension."""

        # data buffers
        self.states = {sense: np.empty((capacity, *shape,), dtype=np.float32) for sense, shape in state_dim.items()}
        self.actions = np.empty((capacity, *action_dim[:-1]), dtype=np.float32 if is_continuous else np.int32)
        self.action_probabilities = np.empty((capacity,), dtype=np.float32)
        self.returns = np.empty((capacity,), dtype=np.float32)
        self.advantages = np.empty((capacity,), dtype=np.float32)
        self.values = np.empty((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)

        # only here to be compatible with TimeDistributed, no functionality in this implementation
        self.mask = np.ones((capacity,), dtype=bool)

        # data not stored but buffered (for postprocessing etc.)
        self.achieved_goals = []

        # secondary data
        self.episode_lengths = []
        self.episode_rewards = []
        self.episodes_completed = 0
        self.auxiliary_performances = {}

        # indicators
        self.is_continuous = is_continuous
        self.capacity = capacity
        self.filled = 0

        self.seq_length = 1

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.filled}/{self.capacity}]"

    def fill(self, s: List[Sensation], a: arr, ap: arr, adv: arr, ret: arr, v: arr, dones: arr, achieved_goals: arr):
        """Fill (and thereby overwrite) the buffer with a 5-tuple of experience."""
        assert np.all(np.array(list(map(len, [s, a, ap, ret, adv, v]))) == len(s)), f"Inconsistent input sizes: {np.array(list(map(len, [s, a, ap, ret, adv, v])))}"

        self.capacity = len(adv)

        s = stack_sensations(s)
        self.states = {sense: s[sense] for sense in self.states.keys()}
        self.actions = a
        self.action_probabilities = ap
        self.returns = ret
        self.advantages = adv
        self.values = v
        self.dones = dones
        self.achieved_goals = achieved_goals

        # update this to avoid incompatible sizes in the case of changing capacity
        self.mask = np.ones((self.capacity,), dtype=bool)

    def normalize_advantages(self):
        """Normalize the buffered advantages using z-scores. This requires the sequences to be of equal lengths,
        hence if this is not guaranteed during pushing to the buffer, pad_buffer has to be called first."""
        rank = len(self.advantages.shape)
        assert rank in [1, 2], f"Illegal rank of advantage tensor, should be 1 or 2 but is {rank}."

        mean = np.mean(self.advantages)
        std = np.maximum(np.std(self.advantages), 1e-6)
        self.advantages = (self.advantages - mean) / std


class TimeSequenceExperienceBuffer(ExperienceBuffer):
    """Experience Buffer for TimeSequence Data

    Attributes:
        states:         dims (B,
    """

    def __init__(self, capacity: int, state_dim: Dict[str, Tuple[int]], action_dim: Tuple[int], is_continuous: bool,
                 seq_length: int):

        super().__init__(capacity, state_dim, action_dim, is_continuous)
        self.seq_length = seq_length

        self.states = {sense: np.zeros((1, capacity, self.seq_length, *shape,), dtype=np.float32) for sense, shape in
                       state_dim.items()}
        self.actions = np.zeros((1, capacity, self.seq_length, *action_dim[:-1],), dtype=np.float32 if is_continuous else np.int32)
        self.action_probabilities = np.zeros((1, capacity, self.seq_length,), dtype=np.float32)
        self.returns = np.zeros((1, capacity, self.seq_length,), dtype=np.float32)
        self.advantages = np.zeros((1, capacity, self.seq_length,), dtype=np.float32)
        self.values = np.zeros((1, capacity, self.seq_length,), dtype=np.float32)

        self.true_number_of_transitions = 0
        self.number_of_subsequences_pushed = 0

        self.dones = np.zeros((1, capacity, self.seq_length,), dtype=bool)
        self.mask = np.zeros(self.advantages.shape, dtype=bool)

        self.last_advantage_stop = 0

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.filled}/{self.capacity} - {self.seq_length}]"

    def push_seq_to_buffer(self, states: List[Sensation], actions: List[arr], action_probabilities: List[arr],
                           values: List[arr], episode_ended: bool = False):
        """Push a sequence to the buffer, constructed from given lists of values."""
        assert np.all(np.array([len(states), len(actions), len(action_probabilities), len(values)]) == len(states)), \
            "Inconsistent input sizes."

        seq_length = len(actions)

        if seq_length < self.seq_length and not episode_ended:
            raise Exception("An incomplete subsequence has been pushed to the buffer without flagging the episode"
                            "as finished during the subsequence.")

        states = stack_sensations(states)
        for sense, sense_buffer in self.states.items():
            sense_buffer[0, self.number_of_subsequences_pushed, :seq_length, ...] = states[sense]

        self.actions[0, self.number_of_subsequences_pushed, :seq_length, ...] = np.stack(actions)
        self.action_probabilities[0, self.number_of_subsequences_pushed, :seq_length] = action_probabilities
        self.values[0, self.number_of_subsequences_pushed, :seq_length] = values

        if episode_ended:
            self.dones[0, self.number_of_subsequences_pushed, seq_length - 1] = 1

        self.number_of_subsequences_pushed += 1
        self.filled += self.actions.shape[1]
        self.true_number_of_transitions += seq_length

    def push_adv_ret_to_buffer(self, advantages: arr, returns: arr):
        """Push advantages and returns of a whole episode."""
        overhang = len(advantages) % self.seq_length

        # split advantages if necessary
        if overhang == 0:
            advantage_chunks = np.split(advantages, indices_or_sections=len(advantages) // self.seq_length)
            return_chunks = np.split(returns, indices_or_sections=len(advantages) // self.seq_length)
        elif len(advantages) > self.seq_length:
            advantage_chunks = (np.split(advantages[:-overhang], indices_or_sections=len(advantages) // self.seq_length)
                                + [advantages[-overhang:]])
            return_chunks = (np.split(returns[:-overhang], indices_or_sections=len(advantages) // self.seq_length)
                             + [returns[-overhang:]])
        else:
            advantage_chunks = [advantages]
            return_chunks = [returns]

        # fill in the subsequences one by on
        for adv_sub_seq, ret_sub_seq in zip(advantage_chunks, return_chunks):
            seq_length = len(adv_sub_seq)
            self.advantages[0, self.last_advantage_stop, :seq_length] = adv_sub_seq
            self.mask[0, self.last_advantage_stop, :seq_length] = np.ones(adv_sub_seq.shape)
            self.returns[0, self.last_advantage_stop, :seq_length] = ret_sub_seq

            self.last_advantage_stop += 1

    def normalize_advantages(self):
        """Normalize the buffered advantages using z-scores. This requires the sequences to be of equal lengths."""
        rank = len(self.advantages.shape)
        assert rank in [1, 2, 3], f"Illegal rank of advantage tensor, should be 1 or 2 but is {rank}."

        # TODO could better keep track of already masked array in self.advantages?
        masked_advantages = np.ma.masked_array(self.advantages, 1 - self.mask)
        mean = masked_advantages.mean()
        std = np.maximum(masked_advantages.std(), 1e-6)
        self.advantages = (self.advantages - mean) / std
