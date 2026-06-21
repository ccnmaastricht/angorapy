"""Unit tests for :mod:`angorapy.common.activations` — the LiF activation.

``lif`` / ``LiF`` implement a soft (smooth) leaky integrate-and-fire firing-rate
nonlinearity, a biologically-motivated activation. Being a *custom* op rather
than a stock Keras layer, it has no upstream test coverage — its correctness is
entirely this project's responsibility, hence direct tests for:

* element-wise shape/dtype preservation (so it drops into any model);
* numerical stability across the full input range (the implementation has
  explicit ``tf.where`` guards to avoid ``NaN``/``inf`` in the large/invalid
  regimes; these must actually hold);
* the defining semantic property — monotonicity (more input current => higher
  firing rate);
* layer/function equivalence and ``get_config``/``from_config`` serialization
  (required for model save-load and ``clone_model``).
"""
import numpy as np
import tensorflow as tf

from angorapy.common.activations import LiF, lif


def test_lif_preserves_shape_and_dtype():
    """The activation is element-wise: output shape and dtype match the input.

    Property
        ``lif(x)`` has the same shape and dtype as ``x``.

    Rationale
        An activation must be a drop-in element-wise op to compose with arbitrary
        layers; this catches shape- or dtype-altering regressions.
    """
    x = tf.constant([[0.5, 1.0, 2.0, 5.0]], dtype=tf.float32)
    out = lif(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_lif_outputs_are_finite():
    """Outputs stay finite across the full input range, including extreme regimes.

    Property
        For inputs spanning the strongly-negative, near-zero, and large-positive
        regimes, all outputs are finite (no ``NaN``/``inf``).

    Rationale
        The implementation uses two-stage ``tf.where`` guards specifically to
        avoid ``NaN`` from ``log1p(exp(.))`` overflow and from the invalid
        branch. This verifies those guards genuinely hold end-to-end.
    """
    x = tf.constant([[-50.0, -20.0, 0.0, 1.0, 30.0, 100.0]], dtype=tf.float32)
    out = lif(x).numpy()
    assert np.all(np.isfinite(out))


def test_lif_is_monotonically_increasing_for_positive_drive():
    """Higher input current yields a strictly higher firing-rate response.

    Property
        On a strictly increasing positive input sequence the output is strictly
        increasing (``diff > 0``).

    Rationale
        Monotonicity (more drive => higher rate) is the defining behaviour of the
        activation; losing it would make the function semantically wrong even if
        it remained finite.
    """
    x = tf.constant([0.5, 1.0, 2.0, 5.0, 10.0], dtype=tf.float32)
    out = lif(x).numpy()
    assert np.all(np.diff(out) > 0)


def test_lif_layer_matches_function():
    """The ``LiF`` layer reproduces the functional ``lif`` implementation.

    Property
        With default hyperparameters, ``LiF()(x)`` equals ``lif(x)``.

    Rationale
        Ensures the layer stays a faithful wrapper and the two implementations do
        not drift apart over time.
    """
    x = tf.constant([[0.1, 1.5, 4.0]], dtype=tf.float32)
    layer = LiF()
    assert np.allclose(layer(x).numpy(), lif(x).numpy())


def test_lif_layer_config_roundtrip():
    """``get_config`` exposes the hyperparameters and a layer can be rebuilt from it.

    Property
        ``get_config`` contains the four hyperparameters, and a layer rebuilt via
        ``LiF.from_config(config)`` produces the same output as the original.

    Rationale
        ``get_config``/``from_config`` is what Keras uses for model save-load and
        ``clone_model``; a broken config silently breaks serialization (the same
        class of failure seen with non-serializable initializer arguments).
    """
    layer = LiF(tau_rc=0.01, tau_ref=0.002, v_th=2.0, gamma=0.05)
    config = layer.get_config()
    for key in ("v_th", "tau_ref", "tau_rc", "gamma"):
        assert key in config

    rebuilt = LiF.from_config(config)
    x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    assert np.allclose(rebuilt(x).numpy(), layer(x).numpy())
