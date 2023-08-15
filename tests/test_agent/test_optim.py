import numpy as np
import tensorflow as tf

from angorapy import make_env
from angorapy.common.policies import BetaPolicyDistribution
from angorapy.models import build_shadow_v2_brain_models, build_wider_models
from angorapy.utilities.model_utils import reset_states_masked, reset_states_masked_tf
from angorapy.utilities.util import flatten


def test_model_state_reset():
    # keras model with 3 fully connected, two LSTM layers, stateful
    env = make_env("LunarLanderContinuous-v2")
    model, _, _ = build_wider_models(env, BetaPolicyDistribution(env), bs=5)

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

