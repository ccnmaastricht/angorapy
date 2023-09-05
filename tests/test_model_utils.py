import numpy as np
import tensorflow as tf

from angorapy.utilities.model_utils import reset_states_masked

import pytest


def test_masked_state_reset():
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
