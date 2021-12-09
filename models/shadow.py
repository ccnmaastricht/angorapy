#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os

import gym
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as TD
# from tensorflow_core.python.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import plot_model

from common.policies import BasePolicyDistribution, BetaPolicyDistribution, MultiCategoricalPolicyDistribution
from common.wrappers import make_env
from models import _build_encoding_sub_model
from models.convolutional import _build_openai_encoder, _build_openai_small_encoder
from common.const import VISION_WH
from utilities.util import env_extract_dims


def build_ssc_module(batch_and_sequence_shape, somatosensation_input_shape):
    """Build the model of the SSC."""
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + somatosensation_input_shape, name="SSCInput")
    ssc = tf.keras.layers.Dense(128)(ssc_input)
    ssc = tf.keras.layers.ReLU()(ssc)
    ssc = tf.keras.layers.Dense(64)(ssc)
    ssc = tf.keras.layers.ReLU()(ssc)

    return tf.keras.Model(inputs=ssc_input, outputs=ssc, name="SomatosensoryCortex")


def build_ppc_module(batch_and_sequence_shape, vc_input_shape, ssc_input_shape):
    """Build the PPC model."""
    vc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + vc_input_shape, name="VCInput")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSCInput")

    spl_input = tf.keras.layers.Concatenate()([vc_input, ssc_input])
    spl = tf.keras.layers.Dense(256)(spl_input)
    spl = tf.keras.layers.ReLU()(spl)

    ipl = tf.keras.layers.Dense(256)(spl)
    ipl = tf.keras.layers.ReLU()(ipl)

    ips_input = tf.keras.layers.Concatenate()([ipl, ssc_input])
    ips = tf.keras.layers.Dense(128)(ips_input)
    ips = tf.keras.layers.ReLU()(ips)

    return tf.keras.Model(inputs=[vc_input, ssc_input], outputs=[spl, ipl, ips], name="PosteriorParietalCortex")


def build_pfc_module(batch_and_sequence_shape, goal_input_shape, ssc_input_shape, it_input_shape):
    """Build the PFC model. Returns output of MCC, LPFC."""

    goal_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + goal_input_shape, name="Goal Input")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSC Input")
    it_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + it_input_shape, name="IT Input")

    mcc_input = tf.keras.layers.Concatenate()([goal_input, ssc_input])
    mcc = tf.keras.layers.Dense(64)(mcc_input)
    mcc = tf.keras.layers.ReLU()(mcc)

    lpfc_input = tf.keras.layers.Concatenate()([mcc, goal_input, it_input])
    lpfc = tf.keras.layers.Dense(128)(lpfc_input)
    lpfc = tf.keras.layers.ReLU()(lpfc)

    return tf.keras.Model(inputs=[goal_input, ssc_input, it_input], outputs=[mcc, lpfc], name="PrefrontalCortex")


def build_mc_module(batch_and_sequence_shape, mcc_input_shape, lpfc_input_shape, ipl_input_shape,
                    ips_input_shape, ssc_input_shape):
    mcc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + mcc_input_shape, name="Goal Input")
    lpfc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + lpfc_input_shape, name="Goal Input")
    ipl_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ipl_input_shape, name="Goal Input")
    ips_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ips_input_shape, name="Goal Input")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="Goal Input")

    pmc_input = tf.keras.layers.Concatenate([lpfc_input, ipl_input])
    pmc = tf.keras.layers.Dense(512)(pmc_input)
    pmc = tf.keras.layers.ReLU()(pmc)
    pmc, *_ = tf.keras.layers.LSTM(512,
                                   stateful=True,
                                   return_sequences=True,
                                   batch_size=batch_and_sequence_shape[0],
                                   return_state=True,
                                   name="pmc_recurrent_layer")(pmc)

    m1_input = tf.keras.layers.Concatenate([pmc, mcc_input, lpfc_input, ssc_input, ips_input])
    m1 = tf.keras.layers.Dense(256)(m1_input)
    m1 = tf.keras.layers.ReLU()(m1)

    return tf.keras.Model(inputs=[mcc_input, lpfc_input, ipl_input, ips_input, ssc_input],
                          outputs=m1, name="MotorCortex")


def build_shadow_brain_base(env: gym.Env, distribution: BasePolicyDistribution, bs: int, model_type: str = "rnn",
                            blind: bool = False, sequence_length=1, **kwargs):
    """Build network for the shadow hand task, version 2."""
    state_dimensionality, n_actions = env_extract_dims(env)

    rnn_choice = {"rnn": tf.keras.layers.SimpleRNN,
                  "lstm": tf.keras.layers.LSTM,
                  "gru": tf.keras.layers.GRU}[
        model_type]

    # inputs
    if blind:
        visual_input = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["vision"],),
                                      name="vision")
    else:
        raise NotImplementedError("Currently Visual input is not Implemented")

    proprio_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["proprioception"],),
                                name="proprioception")
    touch_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["somatosensation"],),
                              name="somatosensation")
    goal_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["goal"],), name="goal")

    input_list = [visual_input, proprio_in, touch_in, goal_in]

    somatosensation = tf.keras.layers.Concatenate()([proprio_in, touch_in])
    somatosensation_masked = tf.keras.layers.Masking(
        batch_input_shape=(bs, sequence_length,) + somatosensation.shape
    )(somatosensation)

    vision = tf.keras.layers.Masking(batch_input_shape=(bs, sequence_length,) + visual_input.shape)(visual_input)

    # sensory cortices
    ssc_model = build_ssc_module((bs * sequence_length,), somatosensation.shape)
    ssc_output = TD(ssc_model, name="SSC")(somatosensation_masked)
    ssc_output.set_shape([bs] + ssc_output.shape[1:])

    # higher cortices
    ppc_model = TD(build_ppc_module((bs, sequence_length,), vision.shape, ssc_output.shape))
    ipl_output, ips_output = ppc_model([vision, ssc_output])

    pfc_model = TD(build_pfc_module((bs, sequence_length,), goal_in.shape, ssc_output.shape, vision.shape))
    mcc_output, lpfc_output = pfc_model([goal_in, ssc_output, vision])

    mc_model = build_mc_module((bs, sequence_length,), mcc_output.shape, lpfc_output.shape, ipl_output.shape,
                               ips_output.shape, ssc_output.shape)
    m1_output = mc_model([mcc_output, lpfc_output, ipl_output, ips_output, ssc_output])

    # output heads
    policy_out = distribution.build_action_head(n_actions, m1_output.shape[1:], bs)(m1_output)
    value_out = tf.keras.layers.Dense(1, name="value")(m1_output)

    # define models
    policy = tf.keras.Model(inputs=input_list, outputs=[policy_out], name="shadow_brain_policy")
    value = tf.keras.Model(inputs=input_list, outputs=[value_out], name="shadow_brain_value")
    joint = tf.keras.Model(inputs=input_list, outputs=[policy_out, value_out],
                           name="shadow_brain" + ("_visual" if not blind else ""))

    return policy, value, joint


def build_shadow_brain_models(env: gym.Env, distribution: BasePolicyDistribution, bs: int, model_type: str = "rnn",
                              blind: bool = False, **kwargs):
    """Build shadow brain networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for broader options to specific functions
    if model_type == "ffn":
        raise NotImplementedError("No non recurrent version of this ShadowBrain abailable.")

    return build_shadow_brain_base(env=env, distribution=distribution, bs=bs, model_type=model_type, blind=blind)


if __name__ == "__main__":
    from environments import *

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf.config.experimental.set_memory_growth("GPU:0", True)

    sequence_length = 4
    batch_size = 256

    env = make_env("HumanoidManipulateBlockDiscrete-v0")
    _, _, joint = build_shadow_brain_base(env, MultiCategoricalPolicyDistribution(env), bs=batch_size, blind=True,
                                          sequence_length=sequence_length)
    plot_model(joint, to_file=f"{joint.name}.png", expand_nested=True, show_shapes=True)
    exit()

    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD()

    joint({
        "vision": tf.random.normal((batch_size, sequence_length, VISION_WH, VISION_WH, 3)),
        "proprioception": tf.random.normal((batch_size, sequence_length, 48)),
        "somatosensation": tf.random.normal((batch_size, sequence_length, 92)),
        "goal": tf.random.normal((batch_size, sequence_length, 15)),
    })

    joint.summary()
