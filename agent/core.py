import collections
from abc import ABC, abstractmethod

import numpy

from learn.exploration import _Explorer


class _RLAgent(ABC):

    def __init__(self):
        self.memory = collections.deque(maxlen=2000)

    def remember(self, experience, done):
        self.memory.append((experience, done))

    @abstractmethod
    def act(self, state: numpy.ndarray, explorer: _Explorer=None):
        pass

    @abstractmethod
    def learn(self, batch_size: int):
        pass
