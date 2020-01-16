"""Custom Exceptions"""

# ENVIRONMENT HANDLING


class UninterpretableObservationSpace(Exception):
    """The environment has an observations space that can currently not be handled by our implementation."""
    pass
