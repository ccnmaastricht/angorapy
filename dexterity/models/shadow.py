#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os

import gym
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as TD
# from tensorflow_core.python.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import plot_model

from dexterity.common.const import VISION_WH
from dexterity.common.policies import BasePolicyDistribution, MultiCategoricalPolicyDistribution
from dexterity.common.wrappers import make_env
from dexterity.utilities.util import env_extract_dims


def build_ssc_module(batch_and_sequence_shape, somatosensation_input_shape):
    """Build the model of the SSC."""
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + somatosensation_input_shape, name="SSCInput")
    ssc = TD(tf.keras.layers.Dense(128))(ssc_input)
    ssc = tf.keras.layers.ReLU()(ssc)
    ssc = TD(tf.keras.layers.Dense(64))(ssc)
    ssc = tf.keras.layers.ReLU()(ssc)

    return tf.keras.Model(inputs=ssc_input, outputs=ssc, name="SomatosensoryCortex")


def build_ppc_module(batch_and_sequence_shape, vc_input_shape, ssc_input_shape):
    """Build the PPC model."""
    vc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + vc_input_shape, name="VCInput")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSCInput")

    spl_input = tf.keras.layers.concatenate([vc_input, ssc_input])
    spl = TD(tf.keras.layers.Dense(256, name="SPL"))(spl_input)
    spl = tf.keras.layers.ReLU()(spl)

    ipl = TD(tf.keras.layers.Dense(256, name="IPL"))(spl)
    ipl = tf.keras.layers.ReLU()(ipl)

    ips_input = tf.keras.layers.concatenate([ipl, ssc_input])
    ips = TD(tf.keras.layers.Dense(128, name="IPS"))(ips_input)
    ips = tf.keras.layers.ReLU()(ips)

    return tf.keras.Model(inputs=[vc_input, ssc_input], outputs=[spl, ipl, ips], name="PosteriorParietalCortex")


def build_pfc_module(batch_and_sequence_shape, goal_input_shape, ssc_input_shape, it_input_shape):
    """Build the PFC model. Returns output of MCC, LPFC."""

    goal_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + goal_input_shape, name="Goal Input")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSC Input")
    it_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + it_input_shape, name="IT Input")

    mcc_input = tf.keras.layers.concatenate([goal_input, ssc_input])
    mcc = TD(tf.keras.layers.Dense(64))(mcc_input)
    mcc = tf.keras.layers.ReLU()(mcc)

    lpfc_input = tf.keras.layers.concatenate([mcc, goal_input, it_input])
    lpfc = TD(tf.keras.layers.Dense(128))(lpfc_input)
    lpfc = tf.keras.layers.ReLU()(lpfc)

    return tf.keras.Model(inputs=[goal_input, ssc_input, it_input], outputs=[mcc, lpfc], name="PrefrontalCortex")


def build_mc_module(batch_and_sequence_shape, mcc_input_shape, lpfc_input_shape, ipl_input_shape,
                    ips_input_shape, ssc_input_shape, rnn_class=tf.keras.layers.LSTM):
    mcc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + mcc_input_shape, name="GoalInput")
    lpfc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + lpfc_input_shape, name="LPFCInput")
    ipl_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ipl_input_shape, name="IPLInput")
    ips_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ips_input_shape, name="IPSInput")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSCInput")

    pmc_input = tf.keras.layers.concatenate([lpfc_input, ipl_input])
    pmc = TD(tf.keras.layers.Dense(512))(pmc_input)
    pmc = tf.keras.layers.ReLU()(pmc)
    pmc, *_ = rnn_class(512,
                        stateful=True,
                        return_sequences=True,
                        batch_size=batch_and_sequence_shape[0],
                        return_state=True,
                        name="pmc_recurrent_layer")(pmc)

    m1_input = tf.keras.layers.concatenate([pmc, mcc_input, lpfc_input, ssc_input, ips_input])
    m1 = TD(tf.keras.layers.Dense(256))(m1_input)
    m1 = tf.keras.layers.ReLU()(m1)

    return tf.keras.Model(inputs=[mcc_input, lpfc_input, ipl_input, ips_input, ssc_input],
                          outputs=[pmc, m1], name="MotorCortex")


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
    goal_input = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["goal"],), name="goal")

    input_list = [visual_input, proprio_in, touch_in, goal_input]

    somatosensation = tf.keras.layers.concatenate([proprio_in, touch_in])
    somatosensation_masked = tf.keras.layers.Masking(
        batch_input_shape=somatosensation.shape
    )(somatosensation)

    vision_masked = tf.keras.layers.Masking(batch_input_shape=visual_input.shape)(visual_input)
    goal_masked = tf.keras.layers.Masking(batch_input_shape=goal_input.shape)(goal_input)

    ssc = build_ssc_module((bs, sequence_length,), (somatosensation_masked.shape[-1],))
    ssc_out = ssc(somatosensation_masked)

    ppc = build_ppc_module((bs, sequence_length,), (vision_masked.shape[-1],), (ssc.output_shape[-1],))
    spl_out, ipl_out, ips_out = ppc([vision_masked, ssc_out])

    pfc = build_pfc_module((bs, sequence_length,), (goal_masked.shape[-1],), (ssc.output_shape[-1],),
                           (vision_masked.shape[-1],))
    mcc_out, lpfc_out = pfc([goal_masked, ssc_out, vision_masked])

    mc = build_mc_module((bs, sequence_length,), (mcc_out.shape[-1],), (lpfc_out.shape[-1],), (ipl_out.shape[-1],),
                         (ips_out.shape[-1],), (ssc_out.shape[-1],), rnn_class=rnn_choice)
    pmc_out, m1_out = mc([mcc_out, lpfc_out, ipl_out, ips_out, ssc_out])

    # policy head
    policy_out = distribution.build_action_head(n_actions, m1_out.shape[1:], bs)(m1_out)

    # value head
    value_inputs = [pmc_out, mcc_out, ips_out, lpfc_out, ssc_out]
    if "asynchronous" in state_dimensionality.keys():
        asynchronous = tf.keras.Input(batch_shape=(bs, sequence_length,) + state_dimensionality["asynchronous"],
                                      name="asynchronous")
        input_list.append(asynchronous)
        value_inputs.append(asynchronous)

    value_in = tf.keras.layers.concatenate(value_inputs)
    value_out = tf.keras.layers.Dense(512)(value_in)
    value_out = tf.keras.layers.Activation("relu")(value_out)
    value_out = tf.keras.layers.Dense(512)(value_out)
    value_out = tf.keras.layers.Activation("relu")(value_out)
    value_out = tf.keras.layers.Dense(1, name="value")(value_out)

    # define models
    policy = tf.keras.Model(inputs=input_list, outputs=[policy_out], name="shadow_brain_policy")
    value = tf.keras.Model(inputs=input_list, outputs=[value_out], name="shadow_brain_value")
    joint = tf.keras.Model(inputs=input_list, outputs=[policy_out, value_out],
                           name="shadow_brain" + ("_visual" if not blind else ""))

    return policy, value, joint


def build_shadow_brain_models(env: gym.Env, distribution: BasePolicyDistribution, bs: int, model_type: str = "lstm",
                              blind: bool = False, **kwargs):
    """Build shadow brain networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for broader options to specific functions
    if model_type == "ffn":
        raise NotImplementedError("No non recurrent version of this ShadowBrain abailable.")

    return build_shadow_brain_base(env=env, distribution=distribution, bs=bs, model_type=model_type, blind=blind)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf.config.experimental.set_memory_growth("GPU:0", True)

    batch_size = 256
    sequence_length = 16

    env = make_env("HumanoidManipulateBlockDiscreteAsynchronous-v0")
    _, _, joint = build_shadow_brain_base(env, MultiCategoricalPolicyDistribution(env), bs=batch_size, blind=True,
                                          sequence_length=sequence_length, model_type="gru")
    plot_model(joint, to_file=f"{joint.name}.png", expand_nested=True, show_shapes=True)

    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD()

    joint({
        "vision": tf.random.normal((batch_size, sequence_length, 7)),
        "proprioception": tf.random.normal((batch_size, sequence_length, 48)),
        "somatosensation": tf.random.normal((batch_size, sequence_length, 92)),
        "asynchronous": tf.random.normal((batch_size, sequence_length, 25)),
        "goal": tf.random.normal((batch_size, sequence_length, 4)),
    })
