from angorapy.tasks.world_building.base import PhysicalWorld
from angorapy.tasks.world_building.entities import Stage
from angorapy.tasks.envs.dexterity.mujoco_model.externals import Block
from angorapy.tasks.envs.dexterity.mujoco_model.robot import ShadowHand


class ShadowHandWithCubeWorld(PhysicalWorld):

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
    world = ShadowHandWithCubeWorld()
    print(world.stage.mjcf_model.to_xml_string())