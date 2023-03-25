#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os

import gym
import keras_cortex
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as TD
# from tensorflow_core.python.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import plot_model

from angorapy.common.activations import LiF, lif
from angorapy.common.const import VISION_WH
from angorapy.common.policies import BasePolicyDistribution, MultiCategoricalPolicyDistribution
from angorapy.common.wrappers import make_env
from angorapy.models import _build_openai_encoder
from angorapy.utilities.util import env_extract_dims


class OrthogonalMP(tf.keras.initializers.Orthogonal):

    def __call__(self, shape, dtype=None):
        return super().__call__(shape, dtype=dtype)


def build_ssc_module(batch_and_sequence_shape, touch_input_shape, activation=tf.keras.layers.ReLU):
    """Build the model of the SSC."""

    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + touch_input_shape, name="SSCInput")
    ssc = TD(tf.keras.layers.Dense(296, name="SSC_1"), name="TD_ssc_1")(ssc_input)
    ssc = activation(name="SSC_activation_1")(ssc)
    ssc = TD(tf.keras.layers.Dense(296, name="SSC_2"), name="TD_ssc_2")(ssc)
    ssc = activation(name="SSC_activation_2")(ssc)

    return tf.keras.Model(inputs=ssc_input, outputs=ssc, name="SomatosensoryCortex")


def build_ppc_module(batch_and_sequence_shape, vc_input_shape, ssc_input_shape, activation=tf.keras.layers.ReLU):
    """Build the PPC model."""
    vc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + vc_input_shape, name="VCInput")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSCInput")

    spl = TD(tf.keras.layers.Dense(456, name="SPL"), name="TD_SPL")(ssc_input)
    spl = activation(name="SPL_activation")(spl)

    ipl_input = tf.keras.layers.concatenate([ssc_input, vc_input, spl])
    ipl = TD(tf.keras.layers.Dense(672, name="IPL"), name="TD_IPL")(ipl_input)
    ipl = activation(name="IPL_activation")(ipl)

    ips_input = tf.keras.layers.concatenate([ipl, ssc_input, vc_input, spl, ssc_input])
    ips = TD(tf.keras.layers.Dense(344, name="IPS"), name="TD_IPS")(ips_input)
    ips = activation(name="IPS_activation")(ips)

    return tf.keras.Model(inputs=[vc_input, ssc_input], outputs=[spl, ipl, ips], name="PosteriorParietalCortex")


def build_pfc_module(batch_and_sequence_shape, goal_input_shape, ipl_input_shape, ips_input_shape, activation=tf.keras.layers.ReLU):
    """Build the PFC model. Returns output of MCC, LPFC."""

    goal_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + goal_input_shape, name="Goal Input")
    ipl_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ipl_input_shape, name="IPL Input")
    ips_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ips_input_shape, name="IPS Input")

    mcc = TD(tf.keras.layers.Dense(216, name="MCC"), name="TD_mcc")(goal_input)
    mcc = activation(name="MCC_activation")(mcc)

    lpfc_input = tf.keras.layers.concatenate([mcc, ipl_input, ips_input])
    lpfc = TD(tf.keras.layers.Dense(360, name="LPFC"), name="TD_lpfc")(lpfc_input)
    lpfc = activation(name="LPFC_activation")(lpfc)

    return tf.keras.Model(inputs=[goal_input, ipl_input, ips_input], outputs=[mcc, lpfc], name="PrefrontalCortex")


def build_mc_module(batch_and_sequence_shape, lpfc_input_shape, spl_input_shape,
                    ssc_input_shape, rnn_class=tf.keras.layers.LSTM, activation=tf.keras.layers.ReLU):
    lpfc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + lpfc_input_shape, name="LPFCInput")
    spl_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + spl_input_shape, name="IPLInput")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSCInput")

    pmc_input = tf.keras.layers.concatenate([lpfc_input, spl_input, ssc_input])

    pmc, *_ = rnn_class(256,
                        stateful=True,
                        return_sequences=True,
                        batch_size=batch_and_sequence_shape[0],
                        return_state=True,
                        name="pmc_recurrent_layer",
                        recurrent_initializer=OrthogonalMP(),
                        dtype="float32"
                        )(pmc_input)

    m1_input = tf.keras.layers.concatenate([pmc, ssc_input, spl_input])

    m1 = TD(tf.keras.layers.Dense(96, name="m1"), )(m1_input)
    m1 = activation(name="M1_activation")(m1)

    return tf.keras.Model(inputs=[lpfc_input, spl_input, ssc_input],
                          outputs=[pmc, m1], name="MotorCortex")


def build_shadow_v2_brain_base(env: gym.Env, distribution: BasePolicyDistribution, bs: int = 1, model_type: str = "rnn",
                            blind: bool = False, sequence_length=1, activation=tf.keras.layers.ReLU, **kwargs):
    """Build network for the shadow hand task, version 2."""
    state_dimensionality, n_actions = env_extract_dims(env)

    rnn_choice = {"rnn": tf.keras.layers.SimpleRNN,
                  "lstm": tf.keras.layers.LSTM,
                  "gru": tf.keras.layers.GRU}[model_type]

    # inputs
    visual_input = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["vision"],), name="vision")
    proprio_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["proprioception"],),
                                name="proprioception")
    touch_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["touch"],),
                              name="touch")
    goal_input = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["goal"],), name="goal")

    input_list = [visual_input, proprio_in, touch_in, goal_input]

    touch = tf.keras.layers.concatenate([proprio_in, touch_in])
    touch_masked = tf.keras.layers.Masking(
        batch_input_shape=touch.shape
    )(touch)

    if blind:
        vc = visual_input
    else:
        visual_component = keras_cortex.cornet.CORNetZ(7)
        vc = TD(visual_component, name="visual_component")(visual_input)

    vision_masked = tf.keras.layers.Masking(batch_input_shape=vc.shape)(vc)
    goal_masked = tf.keras.layers.Masking(batch_input_shape=goal_input.shape)(goal_input)

    ssc = build_ssc_module((bs, sequence_length,),
                           (touch_masked.shape[-1],),
                           activation=activation)
    ssc_out = ssc(touch_masked)

    ppc = build_ppc_module((bs, sequence_length,),
                           (vision_masked.shape[-1],),
                           (ssc.output_shape[-1],),
                           activation=activation)
    spl_out, ipl_out, ips_out = ppc([vision_masked, ssc_out])

    pfc = build_pfc_module((bs, sequence_length,),
                           (goal_masked.shape[-1],),
                           (ipl_out.shape[-1],),
                           (ips_out.shape[-1],),
                           activation=activation)
    mcc_out, lpfc_out = pfc([goal_masked, ipl_out, ips_out])

    mc = build_mc_module((bs, sequence_length,),
                         (lpfc_out.shape[-1],),
                         (spl_out.shape[-1],),
                         (ssc_out.shape[-1],), rnn_class=rnn_choice,
                         activation=activation)
    pmc_out, m1_out = mc([lpfc_out, spl_out, ssc_out])

    # policy head
    policy_out = distribution.build_action_head(n_actions, m1_out.shape[1:], bs)(m1_out)

    # value head
    value_inputs = [pmc_out, mcc_out, ips_out, lpfc_out, ssc_out]
    if "asymmetric" in state_dimensionality.keys():
        asymmetric = tf.keras.Input(batch_shape=(bs, sequence_length,) + state_dimensionality["asymmetric"],
                                      name="asymmetric")
        input_list.append(asymmetric)
        value_inputs.append(asymmetric)

    value_in = tf.keras.layers.concatenate(value_inputs)
    value_out = tf.keras.layers.Dense(512)(value_in)
    value_out = activation()(value_out)
    value_out = tf.keras.layers.Dense(512)(value_out)
    value_out = activation()(value_out)
    value_out = tf.keras.layers.Dense(1, name="value", dtype="float32")(value_out)

    # define models
    policy = tf.keras.Model(inputs=input_list, outputs=[policy_out], name="shadow_brain_policy")
    value = tf.keras.Model(inputs=input_list, outputs=[value_out], name="shadow_brain_value")
    joint = tf.keras.Model(inputs=input_list, outputs=[policy_out, value_out],
                           name="shadow_v2" + ("_visual" if not blind else ""))

    return policy, value, joint


def build_shadow_v2_brain_models(env: gym.Env, distribution: BasePolicyDistribution, bs: int = 1, model_type: str = "lstm",
                              blind: bool = False, activation=tf.keras.layers.ReLU, **kwargs):
    """Build shadow brain networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for broader options to specific functions
    if model_type == "ffn":
        raise NotImplementedError("No non recurrent version of this ShadowBrain abailable.")

    return build_shadow_v2_brain_base(env=env, distribution=distribution, bs=bs, model_type=model_type, blind=blind,
                                   activation=activation)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

    batch_size = 16
    sequence_length = 16

    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD()

    # without vision
    env = make_env("HumanoidManipulateBlockDiscreteAsynchronous-v0")
    _, _, joint = build_shadow_v2_brain_base(env, MultiCategoricalPolicyDistribution(env), bs=batch_size, blind=True,
                                          sequence_length=sequence_length, model_type="gru")
    plot_model(joint, to_file=f"{joint.name}.png", expand_nested=True, show_shapes=True)

    joint.summary()

    out = joint({
        "vision": tf.random.normal((batch_size, sequence_length, 7)),
        "proprioception": tf.random.normal((batch_size, sequence_length, 48)),
        "touch": tf.random.normal((batch_size, sequence_length, 92)),
        "asymmetric": tf.random.normal((batch_size, sequence_length, 25)),
        "goal": tf.random.normal((batch_size, sequence_length, 4)),
    })

    # with vision
    env = make_env("HumanoidVisualManipulateBlockDiscreteAsynchronous-v0")
    _, _, joint = build_shadow_v2_brain_base(env, MultiCategoricalPolicyDistribution(env), bs=batch_size, blind=False,
                                          sequence_length=sequence_length, model_type="gru")
    plot_model(joint, to_file=f"{joint.name}.png", expand_nested=True, show_shapes=True)

    out = joint({
        "vision": tf.random.normal((batch_size, sequence_length, VISION_WH, VISION_WH, 3)),
        "proprioception": tf.random.normal((batch_size, sequence_length, 48)),
        "touch": tf.random.normal((batch_size, sequence_length, 92)),
        "asymmetric": tf.random.normal((batch_size, sequence_length, 25)),
        "goal": tf.random.normal((batch_size, sequence_length, 4)),
    })

    joint.summary()
