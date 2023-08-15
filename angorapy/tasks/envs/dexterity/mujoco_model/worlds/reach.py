from angorapy.tasks.world_building.base import PhysicalWorld
from angorapy.tasks.world_building.entities import Stage
from angorapy.tasks.envs.dexterity.mujoco_model.robot import ShadowHand, ShadowHandReach


class ShadowHandReachWorld(PhysicalWorld):

    def __init__(self):
        self._stage = Stage()
        self._robot = ShadowHandReach()

        self._stage.attach(self._robot)

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def robot(self) -> ShadowHand:
        return self._robot


if __name__ == '__main__':
    world = ShadowHandReach()
    print(world.root.to_xml_string())
