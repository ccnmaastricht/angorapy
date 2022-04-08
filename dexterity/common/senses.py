from typing import List

import numpy
import numpy as np
import tensorflow as tf


class Sensation:
    """Wrapper for state representations. Designed for sensory readings, but non-sensory data can be
    encapsulated too. The convention is to refer to any non-sensory data as proprioception, following the intuition
    that it represents inner states."""

    sense_names = ["vision", "somatosensation", "proprioception", "goal", "asynchronous"]

    def __init__(self, vision: np.ndarray = None, proprioception: np.ndarray = None, somatosensation: np.ndarray = None,
                 goal: np.ndarray = None, asynchronous: np.ndarray = None):
        assert not all(s is None for s in (vision, somatosensation, proprioception, goal)), \
            "Sensation requires data for at least on sense (or 'goal')."

        self.vision = vision.astype(np.float32) if vision is not None else None
        self.somatosensation = somatosensation.astype(np.float32) if somatosensation is not None else None
        self.proprioception = proprioception.astype(np.float32) if proprioception is not None else None
        self.goal = goal.astype(np.float32) if goal is not None else None
        self.asynchronous = asynchronous.astype(np.float32) if asynchronous is not None else None

    def __repr__(self):
        return str(self.__dict__)

    # PROPERTIES

    @property
    def shape(self):
        return (99,)

    @property
    def v(self):
        """Shortcut for vision."""
        return self.vision

    @property
    def s(self):
        """Shortcut for somatosensation."""
        return self.somatosensation

    @property
    def touch(self):
        """Shortcut for somatosensation."""
        return self.somatosensation

    @property
    def t(self):
        """Shortcut for somatosensation."""
        return self.somatosensation

    @property
    def p(self):
        """Shortcut for proprioception."""
        return self.proprioception

    @property
    def g(self):
        """Shortcut for goal."""
        return self.goal

    # MAGIC

    def __iter__(self):
        return iter((self.vision, self.somatosensation, self.proprioception, self.goal, self.asynchronous))

    def __len__(self):
        return 5

    def __contains__(self, item):
        return item in self.dict()

    def __getitem__(self, item):
        return self.dict()[item]

    def __setitem__(self, key, value):
        if key in self.__dict__.keys():
            self.__dict__[key] = value
        else:
            raise ValueError(f"{key} is not a sense")

    # OTHER

    def inject_leading_dims(self, time=False):
        """Expand state (inplace) to have a batch and/or time dimension."""
        sense: numpy.ndarray

        for sense, value in self.dict().items():
            if value is None:
                continue

            self[sense] = np.expand_dims(value, axis=(0 if not time else (0, 1)))

    def with_leading_dims(self, time=False):
        """Return a new state with batch and/or time dimension but keep this Sensation as is."""
        sense: numpy.ndarray

        new_sensation = Sensation(**(self.dict()))
        for sense, value in new_sensation.dict().items():
            if value is None:
                continue

            new_sensation[sense] = np.expand_dims(value, axis=(0 if not time else (0, 1)))

        return new_sensation

    def dict(self):
        """Return a dict of the senses and their values."""
        return {k: self.__dict__[k] for k in Sensation.sense_names if self.__dict__[k] is not None}

    def dict_as_tf(self):
        """Return dict as tf tensor."""
        return {k: tf.convert_to_tensor(self.__dict__[k]) for k in Sensation.sense_names if self.__dict__[k] is not None}


def stack_sensations(sensations: List[Sensation]):
    """Stack Sensation objects over a prepended temporal domain."""
    return Sensation(**{
        sense: np.stack([s[sense] for s in sensations], axis=0) for sense in sensations[0].dict().keys()
    })


if __name__ == '__main__':
    ss = stack_sensations([
        Sensation(goal=np.array([1, 2, 3, 4])),
        Sensation(goal=np.array([0, 1, 2, 3])),
        Sensation(goal=np.array([1, 1, 1, 1])),
    ])

    print(ss)
