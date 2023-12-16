import angorapy as ap
import angorapy.tasks.core
import angorapy.tasks.registration
from angorapy.tasks.wrappers import TaskWrapper


def _test_any_task(task: TaskWrapper):
    state = task.reset()
    for _ in range(100):
        state, r, dterm, dtrunc, info = task.step(task.action_space.sample())

        if dtrunc or dterm:
            state = task.reset()


def test_manipulate():
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlock-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlockVisual-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlockAsymmetric-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlockVisualAsymmetric-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlockDiscrete-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlockVisualDiscrete-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlockDiscreteAsymmetric-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ManipulateBlockVisualDiscreteAsymmetric-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("NoisyManipulateBlock-v0"))


def test_reach():
    _test_any_task(angorapy.tasks.registration.make_task("ReachAbsolute-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ReachRelative-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ReachAbsoluteVisual-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ReachRelativeVisual-v0"))


def test_free_reach():
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachAbsolute-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachRelative-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachAbsoluteVisual-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachRelativeVisual-v0"))


def test_cognitive():
    _test_any_task(angorapy.tasks.registration.make_task("HanoiTower-v0"))