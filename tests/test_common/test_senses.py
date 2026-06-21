"""Unit tests for :mod:`angorapy.common.senses` — the ``Sensation`` state container.

``Sensation`` is the universal state representation passed between the
environment, the model, and the experience buffer. It behaves as a dict over a
fixed set of senses (``vision``, ``touch``, ``proprioception``, ``goal``,
``asymmetric``) while also supporting element-wise arithmetic (used by the
running-mean/normalization postprocessors). Much of the pipeline relies on its
contract *implicitly*, so these tests pin the parts most likely to bite:

* construction-time dtype coercion (float32, to avoid TF dtype-mismatch errors);
* the dict view exposing only *present* senses (so model heads do not see
  phantom ``None`` inputs);
* loud rejection of unknown sense keys;
* correct per-sense element-wise ``+``/``-``/``**``;
* leading-axis stacking for sequence/batch assembly.
"""
import numpy as np
import pytest

from angorapy.common.senses import Sensation, stack_sensations


def _proprioceptive(values):
    """Build a proprioception-only ``Sensation`` from a list of values (float32)."""
    return Sensation(proprioception=np.array(values, dtype=np.float32))


def test_construction_casts_to_float32_and_exposes_shortcut():
    """Provided senses are stored as float32 and reachable via their shortcut property.

    Property
        Inputs are coerced to ``float32`` on construction, and the shorthand
        accessor (``.p``) aliases the full attribute (``.proprioception``).

    Rationale
        Dtype is load-bearing: a stray ``float64`` sense triggers dtype-mismatch
        errors deep inside TensorFlow ops. This pins the cast and the aliases.
    """
    s = Sensation(proprioception=np.array([1, 2, 3]))
    assert s.proprioception.dtype == np.float32
    assert np.array_equal(s.p, s.proprioception)


def test_requires_at_least_one_sense():
    """Constructing a ``Sensation`` with no data at all is rejected.

    Property
        ``Sensation()`` (all senses ``None``) raises :class:`AssertionError`.

    Rationale
        A state must carry at least one sense; failing fast at construction
        surfaces the error at its source rather than as a confusing failure
        later in the model.
    """
    with pytest.raises(AssertionError):
        Sensation()


def test_dict_and_membership_only_include_present_senses():
    """The dict view, ``len`` and ``in`` reflect only the senses actually provided.

    Property
        For a proprioception-only state, ``dict()`` has just that key, ``len`` is
        ``1``, and absent senses (``vision``) are *not* members.

    Rationale
        Model heads iterate over present senses; if ``None`` senses leaked into
        the dict view they would create phantom inputs. This enforces the
        "only-present" contract.
    """
    s = _proprioceptive([1, 2])
    assert list(s.dict().keys()) == ["proprioception"]
    assert len(s) == 1
    assert "proprioception" in s
    assert "vision" not in s


def test_setitem_rejects_unknown_sense():
    """Item assignment accepts real senses and rejects unknown keys.

    Property
        ``s["proprioception"] = ...`` updates the sense; ``s["not_a_sense"] = ...``
        raises :class:`ValueError`.

    Rationale
        Guards against silent typos creating bogus senses; an unknown key must
        fail loudly rather than be quietly stored.
    """
    s = _proprioceptive([1, 2])
    s["proprioception"] = np.array([3.0, 4.0], dtype=np.float32)
    assert np.array_equal(s["proprioception"], [3.0, 4.0])

    with pytest.raises(ValueError):
        s["not_a_sense"] = np.array([0.0])


def test_elementwise_arithmetic():
    """``+``, ``-`` and ``**`` operate element-wise per sense and return a new ``Sensation``.

    Property
        For two proprioception states, addition/subtraction are element-wise and
        ``** 2`` squares element-wise, each producing a fresh ``Sensation``.

    Rationale
        The running-mean / normalization postprocessors perform exactly these
        operations on ``Sensation`` objects; broken per-sense arithmetic would
        corrupt the collected statistics.
    """
    a = _proprioceptive([1.0, 2.0])
    b = _proprioceptive([3.0, 4.0])

    assert np.allclose((a + b).proprioception, [4.0, 6.0])
    assert np.allclose((a - b).proprioception, [-2.0, -2.0])
    assert np.allclose((a ** 2).proprioception, [1.0, 4.0])


def test_stack_sensations_adds_leading_axis():
    """``stack_sensations`` stacks matching senses over a new leading (temporal) axis.

    Property
        Stacking three goal-only states of shape ``(4,)`` yields a goal of shape
        ``(3, 4)``, preserving per-element order.

    Rationale
        Sequence/batch assembly depends on correct leading-axis stacking; a wrong
        axis would silently misshape model inputs.
    """
    stacked = stack_sensations([
        Sensation(goal=np.array([1, 2, 3, 4])),
        Sensation(goal=np.array([0, 1, 2, 3])),
        Sensation(goal=np.array([1, 1, 1, 1])),
    ])
    assert stacked.goal.shape == (3, 4)
    assert np.array_equal(stacked.goal[1], [0, 1, 2, 3])
