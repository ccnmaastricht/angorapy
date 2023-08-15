import unittest
from typing import Sequence, Union

import numpy as np
import pytest

import angorapy as ap

from dm_control import mjcf


class _TestTask(ap.tasks.AnthropomorphicEnv):
    def _set_default_reward_function_and_config(self):
        self.reward_function = lambda x, info: 0
        self.reward_config = {}

    def assert_reward_setup(self):
        pass

    def _env_setup(self, initial_state):
        pass

    def _sample_goal(self):
        return np.zeros(0)


class _TestRobot(ap.tasks.world_building.Robot):

    @property
    def joints(self) -> Sequence[mjcf.Element]:
        return []

    @property
    def actuators(self) -> Sequence[mjcf.Element]:
        return []

    def _parse_entity(self) -> None:
        pass

    def _setup_entity(self) -> None:
        pass

    @property
    def root_body(self) -> Union[None, mjcf.Element]:
        return None


def test_minimal_env():
    robot = _TestRobot(mjcf.RootElement())
    task = _TestTask(robot)

    state = task.reset()
    for _ in range(100):
        state, r, dterm, dtrunc, info = task.step(task.action_space.sample())
        print(state)

    assert True



if __name__ == '__main__':
    unittest.main()
