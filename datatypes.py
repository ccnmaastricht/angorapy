from typing import List

import numpy


class Experience:

    def __init__(self, state: numpy.ndarray, action: int, reward: float, observation: numpy.ndarray, next_action: int=None):
        """ Wrapper class for Experiences gained through exploring the environment.
        Essentially named SARS(A) tuples.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.observation = observation
        self.next_action = next_action

    def has_next_action(self):
        return self.next_action is not None
