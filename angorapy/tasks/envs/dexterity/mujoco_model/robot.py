from typing import Sequence

from dm_control import mjcf
from mujoco_utils import mjcf_utils

from angorapy.tasks.world_building.entities import Robot
from angorapy.tasks.envs.dexterity.mujoco_model import consts

HAND_POSITION = (0., 0., 0.)
HAND_QUATERNION = (1, -1, 1, -1)


class ShadowHand(Robot):
    """A ShadowHand robot implementation."""

    def __init__(self,
                 position: Sequence[float] = HAND_POSITION,
                 rotation: Sequence[float] = HAND_QUATERNION,
                 no_wrist_control: bool = False) -> None:
        """Initializes a ShadowHand.

        Args:
            position: The position of the robot. Defaults to (0, 0, 0).
            rotation: The rotation of the robot, as a quaternion. Defaults to (1, -1, 1, -1), palm facing up.
            no_wrist_control: If True, the wrist is fixed and cannot be controlled. Defaults to False.
        """
        self._prefix = "rh_"
        self.no_wrist_control = no_wrist_control
        super().__init__(consts.MODEL_XML, position=position, rotation=rotation)

    def _parse_entity(self) -> None:
        if self.no_wrist_control:
            self._delete_wrist_actuators()

        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")

        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

    def _setup_entity(self):
        pass

    def _delete_wrist_actuators(self):
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
        for actor in actuators:
            if "_WR" in actor.name:
                actor.remove()

    # VISUALIZATION
    def show_touch_sites(self, show: bool = True):
        for site in self.mjcf_model.worldbody.find_all("site"):
            if "T_" in site.name:
                site.rgba[-1] = 1 if show else 0

    def show_palm_site(self, show: bool = True):
        for site in self.mjcf_model.worldbody.find_all("site"):
            if "palm_center_site" in site.name:
                site.rgba[-1] = 1 if show else 0

    # ACCESSORS
    @property
    def root_body(self) -> mjcf.Element:
        return mjcf_utils.safe_find(self._mjcf_root, "body", self._prefix + "forearm")

    @property
    def joints(self) -> Sequence[mjcf.Element]:
        return self._joints

    @property
    def actuators(self) -> Sequence[mjcf.Element]:
        return self._actuators


class ShadowHandReach(ShadowHand):
    """A Shadow Hand E3M5."""

    def _setup_entity(self):
        super()._setup_entity()

        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1]]

        indicator_body = self.mjcf_model.worldbody
        for finger_idx in range(5):
            # note that initial positions need to be non-[0 0 0] or otherwise they cannot for some reason be changed
            # in simulation. wasted 5h of my life to find this out.

            indicator_body.add(
                "site",
                name=f"target{finger_idx}",
                pos=[0.01, 0, 0],
                rgba=colors[finger_idx],
                size=[0.005],
                type="sphere"
            )

            indicator_body.add(
                "site",
                name=f"finger{finger_idx}",
                pos=[0.01, 0, 0],
                rgba=colors[finger_idx],
                size=[0.01],
                type="sphere"
            )


if __name__ == '__main__':
    shadow_hand = ShadowHand(no_wrist_control=True)
