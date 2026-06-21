"""Unit tests for :mod:`angorapy.agent.utils` — return and advantage estimation.

This module covers the numerical primitives that turn a raw reward/value
trajectory into the learning targets consumed by PPO:

* :func:`get_discounted_returns` — the discounted reward-to-go.
* :func:`estimate_advantage` — Generalized Advantage Estimation (GAE) over a
  batch of (possibly multi-episode) trajectories.
* :func:`estimate_episode_advantages` — a faster ``scipy.signal.lfilter``-based
  GAE for a single contiguous episode.

These functions are pure (NumPy / Python only) and deterministic, which makes
them ideal for exact, known-answer testing. They are also *silent* failure
points: an error here does not raise, it merely produces subtly wrong targets
that bias every gradient the agent takes. The tests therefore pin closed-form
expected values derived by hand, and cross-validate the two GAE implementations
against each other.

References
----------
Schulman et al., "High-Dimensional Continuous Control Using Generalized
Advantage Estimation", ICLR 2016 (arXiv:1506.02438).
"""
import numpy as np
import pytest

from angorapy.agent.utils import (
    estimate_advantage,
    estimate_episode_advantages,
    get_discounted_returns,
)


def test_get_discounted_returns_undiscounted():
    """Return-to-go with ``gamma == 1`` equals the plain sum of following rewards.

    Property
        With no discounting, ``G_t = sum_{k>=t} r_k``. For ``[1, 1, 1]`` this is
        ``[3, 2, 1]``.

    Rationale
        This is the simplest closed form of the return and acts as an
        unambiguous tripwire for regressions in the reversed-accumulate
        implementation (e.g. iterating in the wrong direction).
    """
    returns = get_discounted_returns([1, 1, 1], discount_factor=1.0)
    assert returns == [3, 2, 1]


def test_get_discounted_returns_discounted():
    """Return-to-go applies the discount geometrically to future rewards.

    Property
        ``G_t = r_t + gamma * G_{t+1}``. For rewards ``[1, 1, 1]`` and
        ``gamma = 0.5`` this unrolls to ``[1.75, 1.5, 1.0]``.

    Rationale
        Guards the direction and placement of the discount factor; a reversed
        or mis-applied ``gamma`` yields plausible-looking but wrong magnitudes
        that only a fixed-value check detects.
    """
    returns = get_discounted_returns([1, 1, 1], discount_factor=0.5)
    # r2 = 1; r1 = 1 + .5*1 = 1.5; r0 = 1 + .5*1.5 = 1.75
    assert np.allclose(returns, [1.75, 1.5, 1.0])


def test_estimate_advantage_reduces_to_returns_when_values_zero():
    """GAE collapses to the discounted reward-to-go when the value baseline is zero.

    Property
        With ``V(s) = 0`` everywhere and ``gamma = lam = 1``, GAE has no
        baseline to subtract and no bias/variance trade-off to make, so the
        advantage equals the undiscounted return-to-go.

    Rationale
        GAE is hard to verify by inspection. Anchoring it to a regime with a
        known answer gives a trustworthy correctness baseline for the general,
        harder-to-reason-about cases.
    """
    rewards = [1.0, 1.0]
    values = [0.0, 0.0, 0.0]  # one more than rewards (bootstrap for last state)
    terminals = [False, True]

    advantages = estimate_advantage(rewards, values, terminals, gamma=1.0, lam=1.0)

    assert np.allclose(advantages, [2.0, 1.0])


def test_estimate_advantage_resets_at_episode_boundary():
    """A terminal step prevents earlier steps from bootstrapping across the boundary.

    Property
        When several episodes are concatenated in one batch, the advantage of a
        step must only accumulate rewards/values up to the next terminal step;
        it must not leak signal from the following episode.

    Rationale
        Bootstrapping across episode boundaries is a classic, high-impact RL
        bug that quietly inflates targets. This pins the per-step reset that the
        ``t_is_terminal`` flags are responsible for.
    """
    rewards = [1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0, 0.0]
    # second step is terminal -> first two steps form one episode, last step another
    terminals = [False, True, False]

    advantages = estimate_advantage(rewards, values, terminals, gamma=1.0, lam=1.0)

    # step 0 only sees step 1's reward (boundary blocks step 2): 1 + 1 = 2
    # step 1 is terminal: just its own reward = 1
    # step 2 starts fresh: 1
    assert np.allclose(advantages, [2.0, 1.0, 1.0])


def test_estimate_advantage_requires_one_extra_value():
    """``values`` must hold exactly one more entry than ``rewards`` (the bootstrap).

    Property
        GAE needs a value estimate for the state *after* the last reward to
        bootstrap a possibly non-terminal trajectory; violating this contract
        must raise :class:`ValueError`.

    Rationale
        This length contract is easy to break at call sites (off-by-one when
        slicing value sequences). Asserting a loud failure prevents silent
        misalignment of rewards and values.
    """
    with pytest.raises(ValueError):
        estimate_advantage([1.0, 1.0], [0.0, 0.0], [False, False], gamma=0.99, lam=0.95)


def test_episode_advantages_match_full_estimator_without_terminals():
    """The fast single-episode estimator agrees with the general one (no terminals).

    Property
        For a single contiguous episode (no intermediate terminal steps), the
        ``lfilter``-based :func:`estimate_episode_advantages` and the general
        :func:`estimate_advantage` must produce identical advantages.

    Rationale
        Two independent implementations of the same quantity are a strong
        differential test: a regression in *either* surfaces as a disagreement,
        without needing a separately hand-derived oracle.
    """
    rewards = [1.0, 0.5, -1.0, 2.0]
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    gamma, lam = 0.99, 0.95

    fast = estimate_episode_advantages(rewards, values, gamma, lam)
    full = estimate_advantage(rewards, values, [False] * len(rewards), gamma, lam)

    assert np.allclose(fast, full, atol=1e-5)
