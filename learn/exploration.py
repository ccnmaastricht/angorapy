import random
from abc import ABC, abstractmethod

import numpy


class _Explorer(ABC):

    @abstractmethod
    def choose_action(self, q_values):
        pass


class EpsilonGreedyExplorer(_Explorer):

    def __init__(self, eps_init=0.1, eps_decay=0.999, eps_min=0.001):
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.epsilon = eps_init

    def choose_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randrange(len(q_values))
        else:
            return numpy.argmax(q_values)

    def update(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
