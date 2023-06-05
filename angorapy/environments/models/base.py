import abc
from typing import Sequence

from dm_control import mjcf
from mujoco_utils import types as mj_types


class BaseScene(object):

    def __init__(self, base_xml_file: str, name: str):
        self._mjcf_root = mjcf.from_path(str(base_xml_file))
        self._mjcf_root.model = name

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

    @property
    @abc.abstractmethod
    def joints(self) -> Sequence[mj_types.MjcfElement]:
        ...

    @property
    @abc.abstractmethod
    def actuators(self) -> Sequence[mj_types.MjcfElement]:
        ...
