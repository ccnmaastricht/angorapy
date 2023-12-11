import abc
import pathlib
from typing import Optional, Sequence, Union

from dm_control import mjcf


class _Entity(abc.ABC):

    def __init__(self,
                 root: mjcf.RootElement,
                 name: str,
                 position: Optional[Sequence[float]] = (0, 0, 0),
                 rotation: Optional[Sequence[float]] = (0, 0, 0, 0)):
        self._mjcf_root = root
        self._mjcf_root.model = name

        self._parse_entity()
        self._setup_entity()

        if position is not None and self.root_body is not None:
            self.root_body.pos = position

        if rotation is not None and self.root_body is not None:
            self.root_body.quat = rotation

    @abc.abstractmethod
    def _parse_entity(self) -> None:
        ...

    @abc.abstractmethod
    def _setup_entity(self) -> None:
        ...

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    @abc.abstractmethod
    def root_body(self) -> mjcf.Element:
        ...

    def attach(self, other: Union["_Entity", mjcf.Element]) -> mjcf.Element:
        """Attaches another entity to this one."""
        return self.mjcf_model.attach(other.mjcf_model)


class Robot(_Entity):

    def __init__(self,
                 xml_or_mjcf: Union[str, mjcf.RootElement],
                 position: Sequence[float] = (.0, .0, .0),
                 rotation: Sequence[float] = (.0, .0, .0, 0.)):
        if isinstance(xml_or_mjcf, (str, pathlib.Path)):
            mjfc_element = mjcf.from_path(str(xml_or_mjcf))
        elif isinstance(xml_or_mjcf, mjcf.RootElement):
            mjfc_element = xml_or_mjcf
        else:
            raise ValueError(f"xml_or_mjcf must be either a string or a mjcf.RootElement, got {type(xml_or_mjcf)}")

        super().__init__(mjfc_element, "robot", position=position, rotation=rotation)

    @property
    @abc.abstractmethod
    def joints(self) -> Sequence[mjcf.Element]:
        ...

    @property
    @abc.abstractmethod
    def actuators(self) -> Sequence[mjcf.Element]:
        ...


class External(_Entity):

    def __init__(self,
                 root: mjcf.RootElement,
                 name: str,
                 position: Optional[Sequence[float]] = (0, 0, 0),
                 rotation: Optional[Sequence[float]] = (0, 0, 0, 0)):
        super().__init__(root, name, position=position, rotation=rotation)

    @property
    @abc.abstractmethod
    def joints(self) -> Sequence[mjcf.Element]:
        ...


class Stage(_Entity):
    """Stage with lighting."""

    def __init__(self) -> None:
        """Initializes a stage."""
        super().__init__(mjcf.RootElement(), "stage", position=None, rotation=None)

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
        self._mjcf_root.visual.headlight.set_attributes(
            ambient=(0.4, 0.4, 0.4),
            diffuse=(0.8, 0.8, 0.8),
            specular=(0.8, 0.8, 0.8),
        )

        # Dark checkerboard floor.
        self._mjcf_root.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="flat",
            width=256,
            height=256,
            # rgb1=[0.2, 0.3, 0.4],
            # rgb2=[0.1, 0.2, 0.3],
            rgb1=[1, 1, 1],
            rgb2=[1, 1, 1],
        )
        self._mjcf_root.asset.add(
            "material",
            name="grid",
            texture="grid",
            texrepeat=(5, 5),
            texuniform=True,
            reflectance=0.2,
        )
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            type="plane",
            size=(0, 0, 0.05),
            pos=(0, 0, -0.1),
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
