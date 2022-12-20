"""Custom Exceptions"""

# ENVIRONMENT HANDLING


class UninterpretableObservationSpace(Exception):
    """The environment has an n_steps space that can currently not be handled by our implementation."""
    pass


class IncompatibleModelException(Exception):
    """The model applied to the given environment cannot handle its states."""
    pass


class ComponentError(Exception):
    """Component is incompatible or not loadable."""
    pass
