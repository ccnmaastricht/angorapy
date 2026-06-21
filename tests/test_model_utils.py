"""Unit tests for :mod:`angorapy.utilities.model_utils` — recurrent state handling.

:func:`reset_states_masked` selectively zeroes the hidden states of stateful RNN
layers for only the batch entries flagged by a boolean mask, leaving the others
intact. This is the mechanism that lets a single stateful model carry several
independent episodes in parallel across a batch: when one episode ends, only its
slot's state is reset while the rest continue. A bug here silently mixes hidden
state across episode boundaries and corrupts recurrent learning.
"""
import numpy as np
import tensorflow as tf

from angorapy.utilities.model_utils import (
    is_recurrent_model,
    requires_batch_size,
    requires_sequence_length,
    reset_states_masked,
)

import pytest


def test_masked_state_reset():
    """Only the masked batch entries are reset; unmasked entries keep their state.

    Setup
        A stateful two-LSTM model with batch size 7; all hidden states are
        primed to 9, then reset with the mask ``[T, T, F, T, F, F, T]``.

    Property
        * Masked rows (``True``) are zeroed; unmasked rows (``False``) keep the
          value 9.
        * Both stacked LSTM layers are reset consistently (their states match).

    Rationale
        Per-entry masked resets are what enable parallel multi-episode rollouts
        through one stateful model; this pins the exact rows affected and that
        the reset reaches every recurrent layer, not just the first.
    """
    model = tf.keras.Sequential((
        tf.keras.layers.Dense(2, batch_input_shape=(7, None, 2)),
        tf.keras.layers.LSTM(5, stateful=True, name="larry", return_sequences=True),
        tf.keras.layers.LSTM(5, stateful=True, name="harry"))
    )

    l_layer = model.get_layer("larry")
    h_layer = model.get_layer("harry")
    l_layer.reset_states([s.numpy() + 9 for s in l_layer.states])
    h_layer.reset_states([s.numpy() + 9 for s in h_layer.states])

    reset_states_masked(
        [layer for layer in model.submodules if isinstance(layer, tf.keras.layers.RNN)],
        [True, True, False, True, False, False, True]
    )

    assert np.allclose([s.numpy() for s in model.get_layer("larry").states],
                       [s.numpy() for s in model.get_layer("harry").states])
    assert np.allclose([s.numpy() for s in model.get_layer("larry").states], [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [9, 9, 9, 9, 9],
        [0, 0, 0, 0, 0],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
        [0, 0, 0, 0, 0],
    ])


def test_is_recurrent_model():
    """``is_recurrent_model`` detects the presence of any RNN layer in a model.

    Property
        A model containing an LSTM is recurrent; a purely feed-forward (Dense)
        model is not.

    Rationale
        The agent branches on this flag to decide between the feed-forward and
        the truncated-BPTT optimization/stepping paths, and to add a leading time
        axis to inputs; a wrong answer routes data through the wrong machinery.
    """
    recurrent = tf.keras.Sequential([tf.keras.layers.LSTM(3, input_shape=(None, 2))])
    feedforward = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(2,))])

    assert is_recurrent_model(recurrent)
    assert not is_recurrent_model(feedforward)


def test_requires_batch_size_and_sequence_length():
    """The builder-introspection helpers detect ``bs`` / ``sequence_length`` parameters.

    Property
        ``requires_batch_size`` / ``requires_sequence_length`` return ``True`` iff
        the model-builder signature declares a ``bs`` / ``sequence_length``
        parameter, respectively.

    Rationale
        The agent inspects each builder to decide which keyword arguments to pass
        when constructing models; mis-detecting them would raise at build time or
        silently drop a required argument.
    """
    def builder_with_both(env, distribution, bs=1, sequence_length=1):
        return None

    def builder_with_neither(env, distribution):
        return None

    assert requires_batch_size(builder_with_both)
    assert requires_sequence_length(builder_with_both)
    assert not requires_batch_size(builder_with_neither)
    assert not requires_sequence_length(builder_with_neither)
