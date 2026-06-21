"""Unit test for the task base class :class:`angorapy.tasks.AnthropomorphicEnv`.

This verifies the core environment scaffolding in isolation, decoupled from any
concrete robot or reward: a *minimal* task is assembled from a stub robot (no
joints, no actuators) and a trivial zero-reward function, then stepped. The point
is to confirm the base class's lifecycle — MJCF model assembly, ``reset``,
``step``, observation/info construction, and the action-space plumbing — works
end to end for any subclass, so that failures in real tasks can be attributed to
their specific robot/reward rather than to the shared machinery.
"""
import unittest
from typing import Sequence, Union

import numpy as np
import pytest

import angorapy as ap

from dm_control import mjcf


class _TestTask(ap.tasks.AnthropomorphicEnv):
    """Minimal concrete task: a trivial zero reward and no environment-specific setup."""

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
    """Minimal concrete robot: an empty entity with no joints, actuators, or body."""

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
    """A minimal task built on the base class can be reset and stepped repeatedly.

    Property
        Constructing :class:`_TestTask` around a stub :class:`_TestRobot`, then
        running ``reset`` followed by 100 random ``step`` calls, completes without
        error and returns the standard 5-tuple each step.

    Rationale
        Isolates the shared environment lifecycle (model assembly, reset/step,
        observation+info construction, action sampling) from any concrete robot or
        reward, so regressions in the base class are caught here rather than being
        misattributed to a specific task.
    """
    robot = _TestRobot(mjcf.RootElement())
    task = _TestTask(robot)

    state = task.reset()
    for _ in range(100):
        state, r, dterm, dtrunc, info = task.step(task.action_space.sample())
        print(state)

    assert True



if __name__ == '__main__':
    unittest.main()
