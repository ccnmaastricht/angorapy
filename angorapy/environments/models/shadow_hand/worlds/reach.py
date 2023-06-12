from angorapy.environments.models.base import PhysicalWorld, \
    Stage
from angorapy.environments.models.shadow_hand.shadow_hand import ShadowHand


class ShadowHandReachWorld(PhysicalWorld):

    def __init__(self):
        self._stage = Stage()
        self._robot = ShadowHand()

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
