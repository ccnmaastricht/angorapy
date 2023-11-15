"""Module for additional environments as well as registering modified environments."""
from angorapy.tasks import envs
from angorapy.tasks import world_building
from angorapy.tasks.core import AnthropomorphicEnv

__all__ = [
    "AnthropomorphicEnv",
    "world_building",
]
