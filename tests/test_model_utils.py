import unittest

import gym
import numpy as np
import tensorflow as tf

from angorapy.common.policies import BetaPolicyDistribution
from angorapy.models import get_model_builder
from angorapy.utilities.model_utils import reset_states_masked, build_sub_model_from, get_layers_by_names


class UtilTest(unittest.TestCase):

    def test_masked_state_reset(self):
        model = tf.keras.Sequential((
            tf.keras.layers.Dense(2, batch_input_shape=(7, None, 2)),
            tf.keras.layers.LSTM(5, stateful=True, name="larry", return_sequences=True),
            tf.keras.layers.LSTM(5, stateful=True, name="harry"))
        )

        l_layer = model.get_layer("larry")
        h_layer = model.get_layer("harry")
        l_layer.reset_states([s.numpy() + 9 for s in l_layer.states])
        h_layer.reset_states([s.numpy() + 9 for s in h_layer.states])
        reset_states_masked(model, [True, False, False, True, False, False, True])

        self.assertTrue(np.allclose([s.numpy() for s in model.get_layer("larry").states],
                                    [s.numpy() for s in model.get_layer("harry").states]))
        self.assertTrue(np.allclose([s.numpy() for s in model.get_layer("larry").states], [
            [0, 0, 0, 0, 0],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [0, 0, 0, 0, 0],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [0, 0, 0, 0, 0],
        ]))

    # def test_submodeling_from(self):
    #     env = gym.make("LunarLanderContinuous-v2")
    #     full_model, _, _ = get_model_builder("simple", "gru", shared=False)(env, BetaPolicyDistribution(env))
    #     sub_model_from_a = build_sub_model_from(full_model, "beta_action_head")
    #     sub_model_from_b = build_sub_model_from(full_model, "policy_recurrent_layer")
    #
    #     for sub_model_from in [sub_model_from_a, sub_model_from_b]:
    #         layer = get_layers_by_names(sub_model_from, ["beta_action_head"])[0]
    #
    #         input_shape_raw = layer.get_input_shape_at(1)
    #         input_shape_replaced = tuple(v if v is not None else 1 for v in input_shape_raw)
    #
    #         out = sub_model_from(tf.random.normal(input_shape_replaced))