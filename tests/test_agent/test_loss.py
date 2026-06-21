"""Unit tests for :mod:`angorapy.agent.ppo.loss` — the PPO loss components.

Covers the two scalar objectives optimized by the agent:

* :func:`policy_loss` — the clipped surrogate policy objective (negated, for
  minimization), in both the feed-forward and the recurrent (sequence-masked)
  variants.
* :func:`value_loss` — the (optionally clipped) critic regression loss.

The clipping behaviour *is* PPO: a mistake here does not crash, it silently
turns the algorithm into something else (e.g. an unclipped policy gradient), so
each mechanism is pinned to a hand-computed value rather than only smoke-tested.
Each test states the exact arithmetic it expects in an inline comment so the
expected constant is auditable.

The recurrent variants additionally average over a boolean validity ``mask``
that marks real vs. padded timesteps; two tests specifically guard that padded
steps contribute *nothing* to the loss (a regression test for a fill-value bug
in which masked steps were filled with ``1.`` instead of ``0.`` before the
masked mean).

References
----------
Schulman et al., "Proximal Policy Optimization Algorithms", 2017
(arXiv:1707.06347) — Eq. (7) for the clipped surrogate objective.
"""
import numpy as np
import tensorflow as tf

from angorapy.agent.ppo import loss


def test_policy_loss_baseline_is_negative_mean_advantage():
    """Unchanged policy (ratio == 1) reduces the clipped objective to -mean(advantage).

    Property
        With ``action_prob == old_action_prob`` the probability ratio is 1, both
        arguments of the clipped ``max`` coincide, and the (negated) objective is
        simply ``-mean(advantage)``.

    Rationale
        Pins the sign convention (the returned loss is the *negated* objective to
        be minimized) and the mean reduction, independent of any clipping.
    """
    action_prob = tf.constant([0.5, 0.5])
    old_action_prob = tf.constant([0.5, 0.5])  # ratio = exp(0) = 1
    advantage = tf.constant([1.0, 3.0])

    result = loss.policy_loss(action_prob, old_action_prob, advantage,
                              mask=None, clipping_bound=0.2, is_recurrent=False)

    assert np.isclose(result.numpy(), -2.0)


def test_policy_loss_clips_large_ratio_on_positive_advantage():
    """A ratio beyond 1 + clip with positive advantage is clamped, capping the loss.

    Property
        For ratio ``2`` and ``clip = 0.2`` the surrogate takes
        ``max(-ratio*A, -clip(ratio, 0.8, 1.2)*A) = max(-2, -1.2) = -1.2``,
        i.e. the clipped term wins and limits the (already negative) loss.

    Rationale
        Directly exercises the trust-region clamp — the defining mechanism of
        PPO. Without it the loss would be ``-2.0``; the clamp is what keeps the
        policy update conservative.
    """
    action_prob = tf.constant([np.log(2.0)], dtype=tf.float32)  # ratio = exp(ln2 - 0) = 2
    old_action_prob = tf.constant([0.0], dtype=tf.float32)
    advantage = tf.constant([1.0], dtype=tf.float32)

    result = loss.policy_loss(action_prob, old_action_prob, advantage,
                              mask=None, clipping_bound=0.2, is_recurrent=False)

    # max(-2*1, -clip(2,0.8,1.2)*1) = max(-2.0, -1.2) = -1.2
    assert np.isclose(result.numpy(), -1.2)


def test_policy_loss_recurrent_normalizes_by_mask_count():
    """The recurrent policy loss sums over valid steps and divides by their count.

    Property
        In the recurrent branch the per-step losses are summed and normalized by
        the number of valid (``True``) mask entries rather than averaged with
        ``reduce_mean``. With an all-valid mask and ratio 1, the result is
        ``sum(-advantage) / n_valid``.

    Rationale
        The recurrent reduction differs from the feed-forward ``reduce_mean``
        path and is easy to get wrong; this fixes its normalization.
    """
    action_prob = tf.constant([[0.5], [0.5]])
    old_action_prob = tf.constant([[0.5], [0.5]])  # ratio = 1
    advantage = tf.constant([[2.0], [4.0]])
    mask = tf.constant([[True], [True]])

    result = loss.policy_loss(action_prob, old_action_prob, advantage,
                              mask=mask, clipping_bound=0.2, is_recurrent=True)

    # clipped = -advantage = [-2, -4]; sum / count = -6 / 2 = -3
    assert np.isclose(result.numpy(), -3.0)


def test_policy_loss_recurrent_ignores_masked_steps():
    """Padded (masked-out) timesteps must not influence the recurrent policy loss.

    Property
        Only valid steps contribute: for a ``[True, False]`` mask the loss equals
        the single valid step's value (``-advantage / 1``), and perturbing the
        *masked-out* step's advantage leaves the result unchanged.

    Rationale
        Regression test for the fill-value bug where masked steps were filled
        with ``1.`` (the multiplicative identity) before a *summed* mean,
        injecting ``+1`` per padded step. The absolute check (``-2.0``) is the
        guard: under the old behaviour it evaluated to ``-1.0``.
    """
    action_prob = tf.constant([[0.5], [0.5]])
    old_action_prob = tf.constant([[0.5], [0.5]])  # ratio = 1 -> clipped = -advantage
    mask = tf.constant([[True], [False]])

    # only the first (valid) step should count: -advantage / 1 = -2.0
    result = loss.policy_loss(action_prob, old_action_prob, tf.constant([[2.0], [5.0]]),
                              mask=mask, clipping_bound=0.2, is_recurrent=True)
    assert np.isclose(result.numpy(), -2.0)

    # changing only the masked-out step's advantage must leave the loss unchanged
    perturbed = loss.policy_loss(action_prob, old_action_prob, tf.constant([[2.0], [999.0]]),
                                 mask=mask, clipping_bound=0.2, is_recurrent=True)
    assert np.isclose(perturbed.numpy(), result.numpy())


def test_value_loss_recurrent_ignores_masked_steps():
    """Padded (masked-out) timesteps must not influence the recurrent value loss.

    Property
        Only valid steps contribute: for a ``[True, False]`` mask the loss equals
        ``0.5 * (pred - return)^2`` of the single valid step, and perturbing the
        masked-out step's prediction leaves the result unchanged.

    Rationale
        The critic counterpart to the masked policy-loss regression test. Under
        the old ``1.`` fill this evaluated to ``2.5`` instead of the correct
        ``2.0``; the absolute check guards against that returning.
    """
    old_values = tf.constant([[0.0], [0.0]])
    returns = tf.constant([[1.0], [1.0]])
    mask = tf.constant([[True], [False]])

    # only step 0 counts: (3-1)^2 = 4; / 1 * 0.5 = 2.0
    result = loss.value_loss(tf.constant([[3.0], [7.0]]), old_values, returns,
                             mask=mask, clip=False, clipping_bound=0.2, is_recurrent=True)
    assert np.isclose(result.numpy(), 2.0)

    # changing only the masked-out step's prediction must leave the loss unchanged
    perturbed = loss.value_loss(tf.constant([[3.0], [100.0]]), old_values, returns,
                                mask=mask, clip=False, clipping_bound=0.2, is_recurrent=True)
    assert np.isclose(perturbed.numpy(), result.numpy())


def test_value_loss_is_half_mean_squared_error():
    """Unclipped value loss is ``0.5 * mean((prediction - return)^2)``.

    Property
        With clipping disabled the critic loss is half the mean squared error
        between predictions and returns. For predictions ``[2, 0]`` and returns
        ``[0, 0]`` this is ``0.5 * mean([4, 0]) = 1.0``.

    Rationale
        Locks in the loss formula and the ``0.5`` factor, which sets the relative
        scale of the value loss against the policy loss in the combined objective.
    """
    predictions = tf.constant([2.0, 0.0])
    returns = tf.constant([0.0, 0.0])

    result = loss.value_loss(predictions, old_values=tf.constant([0.0, 0.0]), returns=returns,
                             mask=None, clip=False, clipping_bound=0.2, is_recurrent=False)

    # errors = [4, 0]; mean = 2; * 0.5 = 1.0
    assert np.isclose(result.numpy(), 1.0)


def test_value_loss_clipping_penalizes_large_value_moves():
    """Value clipping is pessimistic: predictions far from the old value are penalized.

    Property
        The clipped value loss takes ``max(clipped_error, error)``. A prediction
        far from ``old_values`` is pinned to ``old +/- clip`` *inside* the squared
        error, producing a *larger* loss than the unclipped MSE — not a smaller,
        clamped one.

    Rationale
        The PPO value clip is commonly mis-implemented as a naive clamp that
        reduces the loss. This test contrasts the unclipped loss (``0.0``) with
        the clipped loss (``48.02``) to verify the pessimistic ``max`` semantics.
    """
    predictions = tf.constant([10.0])
    old_values = tf.constant([0.0])
    returns = tf.constant([10.0])

    unclipped = loss.value_loss(predictions, old_values, returns,
                                mask=None, clip=False, clipping_bound=0.2, is_recurrent=False)
    clipped = loss.value_loss(predictions, old_values, returns,
                              mask=None, clip=True, clipping_bound=0.2, is_recurrent=False)

    # unclipped: (10-10)^2 = 0
    assert np.isclose(unclipped.numpy(), 0.0)
    # clipped: value pinned to old+0.2=0.2 -> (0.2-10)^2 = 96.04; * 0.5 = 48.02
    assert np.isclose(clipped.numpy(), 48.02)
    assert clipped.numpy() > unclipped.numpy()
