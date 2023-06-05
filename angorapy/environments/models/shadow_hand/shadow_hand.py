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

from typing import Optional, \
    Sequence

from dm_control import composer
from mujoco_utils import mjcf_utils, \
    types

from angorapy.environments.models.base import BaseScene
from angorapy.environments.models.shadow_hand import consts

_FINGERTIP_OFFSET = 0.026
_THUMBTIP_OFFSET = 0.0275

# Which dofs to add to the forearm.
_DEFAULT_FOREARM_DOFS = ("forearm_tx", "forearm_ty")


class ShadowHand(BaseScene):
    """A Shadow Hand E3M5."""

    def __init__(self,
                 name: Optional[str] = None) -> None:
        """Initializes a ShadowHand.

        Args:
            name: Name of the hand. Used as a prefix in the MJCF name attributes.
            forearm_dofs: Which dofs to add to the forearm.
        """

        super().__init__(consts.RIGHT_SHADOW_HAND_XML, name)

        self._prefix = "rh_"
        self._parse_robot()
        self._setup_robot()

    def _parse_robot(self) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")

        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

    def _setup_robot(self):
        # Add sites to the tips of the fingers.
        fingertip_sites = []
        for tip_name in consts.FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root,
                "body",
                self._prefix + tip_name
            )
            offset = _THUMBTIP_OFFSET if tip_name == "thdistal" else _FINGERTIP_OFFSET
            tip_site = tip_elem.add(
                "site",
                name=tip_name + "_site",
                pos=(0.0, 0.0, offset),
                type="sphere",
                size=(0.004,),
                group=composer.SENSOR_SITES_GROUP,
            )
            fingertip_sites.append(tip_site)
        self._fingertip_sites = tuple(fingertip_sites)

        # Add joint torque sensors.
        joint_torque_sensors = []
        for joint_elem in self._joints:
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                size=(0.001, 0.001, 0.001),
                type="box",
                rgba=(0, 1, 0, 1),
                group=composer.SENSOR_SITES_GROUP,
            )
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            joint_torque_sensors.append(torque_sensor_elem)
        self._joint_torque_sensors = tuple(joint_torque_sensors)

        # Add velocity and force sensors to the actuators.
        actuator_velocity_sensors = []
        actuator_force_sensors = []
        for actuator_elem in self._actuators:
            velocity_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorvel",
                actuator=actuator_elem,
                name=actuator_elem.name + "_velocity",
            )
            actuator_velocity_sensors.append(velocity_sensor_elem)

            force_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorfrc",
                actuator=actuator_elem,
                name=actuator_elem.name + "_force",
            )
            actuator_force_sensors.append(force_sensor_elem)
        self._actuator_velocity_sensors = tuple(actuator_velocity_sensors)
        self._actuator_force_sensors = tuple(actuator_force_sensors)

    # ACCESSORS
    @composer.cached_property
    def root_body(self) -> types.MjcfElement:
        return mjcf_utils.safe_find(self._mjcf_root,
                                    "body",
                                    self._prefix + "forearm")

    @composer.cached_property
    def fingertip_bodies(self) -> Sequence[types.MjcfElement]:
        return tuple(
            mjcf_utils.safe_find(self._mjcf_root,
                                 "body",
                                 self._prefix + name)
            for name in consts.FINGERTIP_BODIES
        )

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        return self._actuators

    @property
    def joint_torque_sensors(self) -> Sequence[types.MjcfElement]:
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_sites

    @property
    def actuator_velocity_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_velocity_sensors

    @property
    def actuator_force_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_force_sensors


class ShadowHandReach(ShadowHand):
    """A Shadow Hand E3M5."""

    def _setup_robot(self):
        super()._setup_robot()

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
