import numpy as np


class ModelComponent(object):
    """Base class for model components."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the component."""
        return self._name

    def load_weights(self, path: str) -> None:
        """Loads weights into the component."""
        raise NotImplementedError

