from angorapy.environments.models.base import PhysicalWorld, \
    Stage
from angorapy.environments.models.externals import Block
from angorapy.environments.models.shadow_hand.shadow_hand import ShadowHand


class ShadowHandWithCube(PhysicalWorld):

    def __init__(self):
        self._stage = Stage()
        self._robot = ShadowHand()
        self._cube = Block()

        self._stage.attach(self._robot)
        cube_attachment_frame = self._stage.attach(self._cube)
        self._cube.add_object_joint(cube_attachment_frame)

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def robot(self) -> ShadowHand:
        return self._robot

    @property
    def cube(self) -> Block:
        return self._cube


if __name__ == '__main__':
    world = ShadowHandWithCube()
    print(world.stage.mjcf_model.to_xml_string())