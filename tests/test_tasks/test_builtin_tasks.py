import angorapy as ap
from angorapy.tasks.wrappers import BaseWrapper


def _test_any_task(task: BaseWrapper):
    state = task.reset()
    for _ in range(100):
        state, r, dterm, dtrunc, info = task.step(task.action_space.sample())


def test_manipulate():
    _test_any_task(ap.make_env("ManipulateBlock-v0"))
    _test_any_task(ap.make_env("ManipulateBlockVisual-v0"))
    _test_any_task(ap.make_env("ManipulateBlockAsymmetric-v0"))
    _test_any_task(ap.make_env("ManipulateBlockVisualAsymmetric-v0"))
    _test_any_task(ap.make_env("ManipulateBlockDiscrete-v0"))
    _test_any_task(ap.make_env("ManipulateBlockVisualDiscrete-v0"))
    _test_any_task(ap.make_env("ManipulateBlockDiscreteAsymmetric-v0"))
    _test_any_task(ap.make_env("ManipulateBlockVisualDiscreteAsymmetric-v0"))

    assert True


def test_reach():
    _test_any_task(ap.make_env("ReachAbsolute-v0"))


def test_free_reach():
    _test_any_task(ap.make_env("FreeReachAbsolute-v0"))


def test_cognitive():
    _test_any_task(ap.make_env("HanoiTower-v0"))