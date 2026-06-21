"""Unit tests for :mod:`angorapy.utilities.core` — general-purpose helpers.

These are small, broadly-used utilities whose silent breakage propagates widely
through the codebase: list flattening of nested model outputs, tiling math for
visualization, dict/observation batching, recurrent reset detection, and the
extraction of environment dimensionalities used to build models. Because they
sit underneath much of the agent and task machinery, each is pinned with an
exact, deterministic expectation.

Also included is a behavioural test for :func:`suppress_type_inference_warning`,
the file-descriptor-redirecting context manager used to silence a benign
TensorFlow grappler warning during recurrent optimization. As it rewires the
process's ``stderr``, it is explicitly checked to (a) be transparent and (b)
never swallow exceptions raised in its body.
"""
import gymnasium as gym
import numpy as np
import tensorflow as tf

from angorapy.utilities.core import (
    detect_finished_episodes,
    env_extract_dims,
    find_divisors,
    find_optimal_tile_shape,
    flatten,
    stack_dicts,
    suppress_type_inference_warning,
)


def test_flatten_nested_list():
    """``flatten`` recursively unnests an arbitrarily nested list into a flat one.

    Property
        ``[1, [2, [3, 4]], 5] -> [1, 2, 3, 4, 5]``.

    Rationale
        ``flatten`` is used across the agent (e.g. to normalize variably-nested
        policy outputs); the recursion must fully unnest regardless of depth.
    """
    assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]


def test_flatten_wraps_non_list_scalar():
    """A non-list scalar is wrapped into a single-element list.

    Property
        ``flatten(7) == [7]``.

    Rationale
        Callers rely on ``flatten`` always returning a list for uniform handling;
        this pins the easily-overlooked scalar base case.
    """
    assert flatten(7) == [7]


def test_find_divisors_returns_all_divisors():
    """``find_divisors`` yields exactly the divisors of the input (no more, no fewer).

    Property
        The set of returned values for ``12`` is ``{1, 2, 3, 4, 6, 12}`` and
        every returned value divides ``12`` evenly.

    Rationale
        Underpins :func:`find_optimal_tile_shape`; a missing divisor would
        eliminate valid tilings and surface as a downstream shape error.
    """
    divisors = find_divisors(12)
    assert sorted(set(divisors)) == [1, 2, 3, 4, 6, 12]
    assert all(12 % d == 0 for d in divisors)


def test_find_optimal_tile_shape():
    """A tiling is found whose dimensions divide the floor and multiply to the tile size.

    Property
        For a ``(4, 6)`` floor and tile size ``12``, the returned ``(h, w)``
        satisfies ``h * w == 12``, ``4 % h == 0`` and ``6 % w == 0``.

    Rationale
        Used to lay out feature maps/observations into grids; an invalid tiling
        either raises or silently misshapes batched tensors.
    """
    hd, wd = find_optimal_tile_shape((4, 6), tile_size=12)
    assert hd * wd == 12
    assert 4 % hd == 0 and 6 % wd == 0


def test_stack_dicts_stacks_matching_keys():
    """``stack_dicts`` stacks the values of matching keys along a new leading axis.

    Property
        Two dicts ``{"a": [1, 2]}`` and ``{"a": [3, 4]}`` stack to
        ``{"a": [[1, 2], [3, 4]]}`` of shape ``(2, 2)``.

    Rationale
        Dict-structured (multi-sense) observations are batched with this helper;
        correct new-axis stacking is required for the data pipeline.
    """
    stacked = stack_dicts([{"a": np.array([1, 2])}, {"a": np.array([3, 4])}])
    assert stacked["a"].shape == (2, 2)
    assert np.array_equal(stacked["a"], [[1, 2], [3, 4]])


def test_detect_finished_episodes_reduces_over_time():
    """An episode is "finished" if *any* step in its ``(B, S)`` done-tensor is True.

    Property
        Reduction is over the time axis: ``[[F, F, T], [F, F, F]] -> [True, False]``.

    Rationale
        Drives recurrent hidden-state resets. Reducing over the wrong axis would
        reset states at the wrong times and corrupt sequence learning.
    """
    dones = tf.constant([[False, False, True], [False, False, False]])
    finished = detect_finished_episodes(dones)
    assert finished.numpy().tolist() == [True, False]


def test_env_extract_dims_discrete_action_space():
    """Box observation + discrete action -> proprioception dims and an ``(n,)`` action dim.

    Property
        ``CartPole-v1`` (obs ``Box(4,)``, action ``Discrete(2)``) maps to
        ``obs_dim == {"proprioception": (4,)}`` and ``act_dim == (2,)``.

    Rationale
        Model input/output shapes are derived from this mapping; the discrete
        ``(n,)`` convention must stay distinct from the continuous one (below).
    """
    obs_dim, act_dim = env_extract_dims(gym.make("CartPole-v1"))
    assert obs_dim == {"proprioception": (4,)}
    assert act_dim == (2,)


def test_env_extract_dims_continuous_action_space():
    """Box observation + box action -> an ``(action_dim, 1)`` action shape.

    Property
        ``Pendulum-v1`` (obs ``Box(3,)``, action ``Box(1,)``) maps to
        ``obs_dim == {"proprioception": (3,)}`` and ``act_dim == (1, 1)``.

    Rationale
        The continuous convention appends a trailing ``1`` (distribution
        parameter axis); drifting from it breaks continuous-control model heads.
    """
    obs_dim, act_dim = env_extract_dims(gym.make("Pendulum-v1"))
    assert obs_dim == {"proprioception": (3,)}
    assert act_dim == (1, 1)


def test_suppress_type_inference_warning_is_transparent_context():
    """The stderr-suppression context is transparent and never swallows exceptions.

    Property
        The context manager executes its body normally and re-raises any
        exception raised inside it (it must not suppress errors along with the
        targeted warning).

    Rationale
        The implementation redirects the C-level ``stderr`` file descriptor
        through a filtering pump thread; a bug there could silently eat real
        errors. This guards both the happy path and exception propagation.
    """
    with suppress_type_inference_warning():
        value = 1 + 1
    assert value == 2

    raised = False
    try:
        with suppress_type_inference_warning():
            raise RuntimeError("boom")
    except RuntimeError:
        raised = True
    assert raised, "context manager must propagate exceptions from its body"
