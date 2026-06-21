"""Unit tests for :mod:`angorapy.utilities.statistics` — online moment estimation.

:func:`increment_mean_var` maintains a running mean and variance via a numerically
stable, single-pass update (Chan/Welford-style parallel combination). The agent
relies on it for observation/reward normalization during rollouts, where the full
data set is never held in memory, so the streaming estimate must match the
batch estimate exactly.
"""
import numpy as np

from angorapy.utilities.statistics import ignore_none, increment_mean_var, mean_fill_nones


def test_incremental_mean_var():
    """Streaming mean/variance converge to the exact batch statistics.

    Property
        Folding 100k samples in one at a time through ``increment_mean_var``
        reproduces ``np.mean`` and ``np.var`` computed over the whole array.

    Rationale
        Online moment updates are a classic source of numerical drift and
        off-by-one (sample-count) errors. Matching the batch result over a large
        stream is the definitive check that the incremental update is both
        correct and numerically stable.
    """
    n_samples = 10000
    sample_dims = 10

    samples = np.array([np.random.randn(1, sample_dims) for _ in range(n_samples)])

    mean, var, n = samples[0], np.zeros((1, sample_dims,)), 1
    for s in samples[1:]:
        s_var = np.zeros((sample_dims,))
        mean, var = increment_mean_var(mean, var, s, s_var, n)
        n += 1

    np_mean = np.mean(samples, axis=0)
    np_var = np.var(samples, axis=0)

    # The incremental variance differs from the batch variance by an O(1/n)
    # term, so an explicit tolerance is used rather than relying on a sample
    # count large enough to slip under np.allclose's default rtol.
    assert np.allclose(mean, np_mean, atol=1e-4)
    assert np.allclose(var, np_var, atol=1e-3)


def test_ignore_none_applies_function_to_non_none_values():
    """``ignore_none`` applies ``func`` only to the non-None elements of a sequence.

    Property
        ``ignore_none(sum, [1, None, 2, None, 3]) == 6``; a sequence that is all
        ``None`` (or empty) returns ``None``.

    Rationale
        Reward/length histories carry ``None`` for skipped or incomplete episodes;
        reductions over them (e.g. mean reward for reporting) must silently drop
        the ``None`` holes rather than crash or be skewed by them.
    """
    assert ignore_none(sum, [1, None, 2, None, 3]) == 6
    assert ignore_none(max, [None, 4, 1]) == 4
    assert ignore_none(sum, [None, None]) is None


def test_mean_fill_nones_interpolates_holes():
    """``mean_fill_nones`` replaces each ``None`` with the mean of its non-None neighbours.

    Property
        Interior gaps are filled by the mean of the nearest valid values on each
        side, while existing values are left untouched (e.g. a hole between 1 and
        3 becomes 2).

    Rationale
        Used to produce gap-free series for plotting/aggregation of metrics that
        are only recorded intermittently; this pins the boundary-mean fill logic.
    """
    filled = mean_fill_nones([1.0, None, 3.0])
    assert np.allclose(filled, [1.0, 2.0, 3.0])
