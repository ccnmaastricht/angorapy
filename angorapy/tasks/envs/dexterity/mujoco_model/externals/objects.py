from pathlib import Path
from typing import Sequence

from dm_control import mjcf

from angorapy.tasks.world_building.entities import External


class Block(External):

    def __init__(self,
                 position: Sequence[float] = (.0, .0, .0),
                 rotation: Sequence[float] = (.0, .0, .0, .0)):
        super().__init__(mjcf.RootElement(),
                         "block",
                         position=position,
                         rotation=rotation)

    def _parse_entity(self) -> None:
        pass

    def _setup_entity(self) -> None:
        self.mjcf_model.compiler.texturedir = str(Path(__file__).parent / "../textures")

        self.mjcf_model.asset.add(
            "texture",
            name="texgeom",
            type="cube",
            builtin="flat",
            mark="cross",
            width=127,
            height=127,
            rgb1=[0.3, 0.6, 0.5],
        )

        self.mjcf_model.asset.add(
            "texture",
            name="texture:object",
            file="block.png",
            gridsize=[3, 4],
            gridlayout=".U..LFRB.D..")

        self.mjcf_model.asset.add(
            "texture",
            name="texture:hidden",
            file="block_hidden.png",
            gridsize=[3, 4],
            gridlayout=".U..LFRB.D..")

        self.mjcf_model.asset.add(
            "material",
            name="object",
            texture="texgeom",
            texuniform=False
        )

        self.mjcf_model.asset.add(
            "material",
            name="material:object",
            texture="texture:object",
            specular=1,
            shininess=0.3,
            reflectance=0
        )

        self.mjcf_model.asset.add(
            "material",
            name="material:hidden",
            texture="texture:hidden",
            specular=1,
            shininess=0.3,
            reflectance=0,
        )
        self.mjcf_model.asset.add(
            "material",
            name="material:target",
            texture="texture:object",
            specular=1,
            shininess=0.3,
            reflectance=0,
            rgba=[1, 1, 1, 0.5]
        )

        self.mjcf_model.worldbody.add("geom", name="object", type="box", size=[0.025, 0.025, 0.025],
                                      material="material:object",
                                      condim=4,
                                      density=567)
        self.mjcf_model.worldbody.add("geom", name="object_hidden", type="box",
                        size=[0.024, 0.024, 0.024],
                        material="material:hidden",
                        condim=4,
                        contype=0,
                        conaffinity=0,
                        mass=0
                        )
        self.mjcf_model.worldbody.add("site",
                                      name="object:center",
                                      pos=[0, 0, 0],
                                      size=[0.01, 0.01, 0.01],
                                      rgba=[1, 0, 0, 0])

    def add_object_joint(self, attachment_frame: mjcf.Element):
        attachment_frame.set_attributes(
            pos=[0.36, 0.0, 0.1]
        )
        attachment_frame.add("joint", type="free", damping=0.01, name="object:joint")

    @property
    def joints(self) -> Sequence[mjcf.Element]:
        return self._mjcf_root.find_all("joint")

    @property
    def root_body(self) -> mjcf.Element:
        return self.mjcf_model.find("body", "block")
