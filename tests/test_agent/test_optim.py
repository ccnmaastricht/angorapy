"""Integration test for recurrent state handling during optimization.

This guards the equivalence of the two stateful-RNN reset implementations used by
the agent — the eager NumPy-based :func:`reset_states_masked` and the graph-mode
:func:`reset_states_masked_tf` — when applied to a real built model. They must
stay byte-for-byte interchangeable: the eager path is used in some code paths and
the ``tf.function`` path in others, and any divergence would make recurrent
training behave differently depending on which path executes.
"""
import numpy as np
import tensorflow as tf

from angorapy import make_task
from angorapy.common.policies import BetaPolicyDistribution
from angorapy.utilities.model_utils import reset_states_masked, reset_states_masked_tf
from angorapy.utilities.core import flatten
from angorapy.models import get_model_builder


def test_model_state_reset():
    """Eager and graph-mode masked resets keep two identical models in lockstep.

    Setup
        Two clones of a stateful LSTM model (identical weights). Over many steps
        they are fed identical inputs and then reset with the *same* random mask
        — one clone via :func:`reset_states_masked` (eager) and the other via
        :func:`reset_states_masked_tf` (graph).

    Property
        * Initially the clones' hidden states are equal.
        * After repeated stepping + masked resets they remain equal, proving the
          two reset implementations are equivalent.
        * The post-reset states differ from the initial states, confirming the
          test actually exercised state changes (not a vacuous pass).

    Rationale
        The agent relies on both reset implementations interchangeably; this
        differential test ensures the graph-mode version faithfully mirrors the
        reference eager version on a real model.
    """
    env = make_task("LunarLanderContinuous-v2")
    model_builder = get_model_builder("simple", model_type="lstm")
    model, _, _ = model_builder(env, BetaPolicyDistribution(env), bs=5)

    # copy of model with same weights
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    model(tf.ones((5, 1, 8)))
    model_copy(tf.ones((5, 1, 8)))

    # store copy of the states of the models
    states = [tf.identity(s) for s in flatten([l.states for l in model.layers if hasattr(l, "states")])]
    states_copy = [tf.identity(s) for s in flatten([l.states for l in model_copy.layers if hasattr(l, "states")])]

    # assert whether the states of the two models are equal
    assert np.all([np.all(s == sc) for s, sc in zip(states, states_copy)])

    recurrent_layers = [layer for layer in model.submodules if isinstance(layer, tf.keras.layers.RNN)]
    recurrent_layers_of_copy = [layer for layer in model_copy.submodules if isinstance(layer, tf.keras.layers.RNN)]

    # reset the states of the models
    for _ in range(10):
        for i in range(3):
            model(tf.ones((5, 1, 8)))
            model_copy(tf.ones((5, 1, 8)))

        mask = tf.cast(tf.random.uniform((5,), minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)
        reset_states_masked(recurrent_layers, mask)
        reset_states_masked_tf(recurrent_layers_of_copy, mask)

    # store the new states of the models
    reset_states = [tf.identity(s) for s in flatten([l.states for l in model.layers if hasattr(l, "states")])]
    reset_states_copy = [tf.identity(s) for s in flatten([l.states for l in model_copy.layers if hasattr(l, "states")])]

    # assert that the states of the models are equal
    assert np.all([np.all(s == sc) for s, sc in zip(reset_states, reset_states_copy)])

    # assert that the new states differ from the old states
    assert not np.all([np.all(s == sc) for s, sc in zip(states, reset_states)])
    assert not np.all([np.all(s == sc) for s, sc in zip(states_copy, reset_states_copy)])
