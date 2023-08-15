import abc

from dm_control import mjcf

from angorapy.tasks.world_building.entities import Robot


class PhysicalWorld(abc.ABC):
    """A physical world in which an environment is simulated."""

    @property
    @abc.abstractmethod
    def robot(self) -> Robot:
        ...

    @property
    @abc.abstractmethod
    def stage(self) -> Robot:
        ...

    @property
    def root(self) -> mjcf.RootElement:
        """Returns the root element of the physical world."""
        return self.stage.mjcf_model
