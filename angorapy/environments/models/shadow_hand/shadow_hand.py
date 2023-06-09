# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shadow hand composer class."""

from typing import Sequence

from dm_control import mjcf
from dm_control.mjcf import Element
from mujoco_utils import mjcf_utils

from angorapy.environments.models.base import Robot
from angorapy.environments.models.shadow_hand import consts

_FINGERTIP_OFFSET = 0.026
_THUMBTIP_OFFSET = 0.0275

HAND_POSITION = (0., 0., 0.)
HAND_QUATERNION = (1, -1, 1, -1)


class ShadowHand(Robot):
    """A ShdaowHand robot implementation."""

    def __init__(self,
                 position: Sequence[float] = HAND_POSITION,
                 rotation: Sequence[float] = HAND_QUATERNION) -> None:
        """Initializes a ShadowHand.

        Args:
            position: The position of the robot. Defaults to (0, 0, 0).
            rotation: The rotation of the robot, as a quaternion. Defaults to (1, -1, 1, -1), palm facing up.
        """
        self._prefix = "rh_"
        super().__init__(consts.RIGHT_SHADOW_HAND_XML, position=position, rotation=rotation)

    def _parse_entity(self) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")

        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

    def _setup_entity(self):
        pass

    # ACCESSORS
    @property
    def root_body(self) -> Element:
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
    shadow_hand = ShadowHand(name="ShadowHand")
