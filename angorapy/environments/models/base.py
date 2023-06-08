import abc
from typing import Sequence

from dm_control import mjcf
from mujoco_utils import types as mj_types


class _Entity(abc.ABC):

    def __init__(self, root: mj_types.MjcfRootElement, name: str):
        self._mjcf_root = root
        self._mjcf_root.model = name

        self._parse_entity()
        self._setup_entity()

    @abc.abstractmethod
    def _parse_entity(self) -> None:
        ...

    @abc.abstractmethod
    def _setup_entity(self) -> None:
        ...

    @property
    def mjcf_model(self) -> mj_types.MjcfRootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    @abc.abstractmethod
    def root_body(self) -> mj_types.MjcfElement:
        ...


class Robot(_Entity):

    def __init__(self, base_xml_file: str):
        super().__init__(mjcf.from_path(str(base_xml_file)), "robot")

    @property
    @abc.abstractmethod
    def joints(self) -> Sequence[mj_types.MjcfElement]:
        ...

    @property
    @abc.abstractmethod
    def actuators(self) -> Sequence[mj_types.MjcfElement]:
        ...


class External(_Entity):

    def __init__(self, base_xml_file: str):
        super().__init__(mjcf.from_path(str(base_xml_file)), "external")


    @property
    @abc.abstractmethod
    def joints(self) -> Sequence[mj_types.MjcfElement]:
        ...


class Stage(_Entity):
    """Stage with lighting."""

    def __init__(self) -> None:
        """Initializes a stage."""
        super().__init__(mjcf.RootElement(), "stage")

    def _parse_entity(self) -> None:
        pass

    def _setup_entity(self) -> None:
        # Change free camera settings.
        self._mjcf_root.statistic.extent = 0.5
        self._mjcf_root.statistic.center = (0.3, 0, 0.0)
        getattr(self._mjcf_root.visual, "global").azimuth = 220
        getattr(self._mjcf_root.visual, "global").elevation = -30

        self._mjcf_root.visual.scale.forcewidth = 0.04
        self._mjcf_root.visual.scale.contactwidth = 0.2
        self._mjcf_root.visual.scale.contactheight = 0.03

        # Lights.
        self._mjcf_root.worldbody.add("light", pos=(0, 0, 1))
        self._mjcf_root.worldbody.add(
            "light", pos=(0.3, 0, 1), dir=(0, 0, -1), directional=False
        )

        # Dark checkerboard floor.
        self._mjcf_root.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=[0.1, 0.1, 0.1],
            rgb2=[0.2, 0.2, 0.2],
        )
        self._mjcf_root.asset.add(
            "material",
            name="grid",
            texture="grid",
            texrepeat=(1, 1),
            texuniform=True,
            reflectance=0.2,
        )
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            type="plane",
            size=(1, 1, 0.05),
            material="grid",
            contype=0,
            conaffinity=0,
        )

        self._mjcf_root.asset.add(
            "texture",
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=[0.2, 0.2, 0.2],
            rgb2=[0.0, 0.0, 0.0],
            width=800,
            height=800,
            mark="random",
            markrgb=[1, 1, 1],
        )

    @property
    def root_body(self) -> mjcf.Element:
        return self.mjcf_model.find("worldbody")
