"""Smoke tests for the built-in registered tasks.

Each test instantiates a family of registered environments via ``make_task`` and
drives them through a short random-action rollout (resetting on termination). The
goal is to confirm every registered variant actually constructs and runs: that
its MJCF model compiles, its observation/action spaces are consistent, and a full
``reset``/``step`` cycle works — across the cartesian product of options each
family exposes (e.g. absolute vs. relative control, vision on/off, discrete vs.
continuous, symmetric vs. asymmetric observations).

These guard against registration drift and per-variant construction errors, which
are easy to introduce when adding options and otherwise only surface when a user
selects that specific variant.
"""
import angorapy as ap
import angorapy.tasks.core
import angorapy.tasks.registration
from angorapy.tasks.wrappers import TaskWrapper


def _test_any_task(task: TaskWrapper):
    """Reset the task and run 100 random steps, resetting whenever an episode ends.

    A generic exerciser: it does not assert specific values, only that an arbitrary
    task can be reset and stepped with sampled actions for many steps (including
    across episode boundaries) without raising.
    """
    state = task.reset()
    for _ in range(100):
        state, r, dterm, dtrunc, info = task.step(task.action_space.sample())

        if dtrunc or dterm:
            state = task.reset()


def test_manipulate():
    """All ManipulateBlock variants construct and run.

    Rationale
        Sweeps the full manipulation matrix (vision, asymmetry, discrete/continuous,
        noisy) to ensure every registered variant builds and steps; these share an
        MJCF model and registration logic where an option can easily break one
        combination.
    """
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
    """All Reach variants (absolute/relative, vision on/off) construct and run.

    Rationale
        Confirms the goal-conditioned reach task builds and steps across its
        control-mode and vision options.
    """
    _test_any_task(angorapy.tasks.registration.make_task("ReachAbsolute-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ReachRelative-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ReachAbsoluteVisual-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("ReachRelativeVisual-v0"))


def test_free_reach():
    """All FreeReach variants (absolute/relative, vision on/off) construct and run.

    Rationale
        Covers the free-reach family (no fixed finger target), guarding its
        distinct goal setup across the same control/vision option grid as Reach.
    """
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachAbsolute-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachRelative-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachAbsoluteVisual-v0"))
    _test_any_task(angorapy.tasks.registration.make_task("FreeReachRelativeVisual-v0"))


def test_cognitive():
    """The cognitive HanoiTower task constructs and runs.

    Rationale
        Covers the non-dexterity, cognitive task family, ensuring its distinct
        (non-MuJoCo) environment also satisfies the reset/step contract.
    """
    _test_any_task(angorapy.tasks.registration.make_task("HanoiTower-v0"))
