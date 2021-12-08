#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os

import gym
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as TD
# from tensorflow_core.python.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import plot_model

from common.policies import BasePolicyDistribution, BetaPolicyDistribution
from common.wrappers import make_env
from models import _build_encoding_sub_model
from models.convolutional import _build_openai_encoder, _build_openai_small_encoder
from common.const import VISION_WH
from utilities.util import env_extract_dims

SSCModule = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.ReLU()],
    name="Somatosensory Cortex"
)


def build_ppc_module(batch_and_sequence_shape, vc_input_shape, ssc_input_shape):
    """Build the PPC model."""
    vc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + vc_input_shape, name="VC Input")
    ssc_input = tf.keras.Input(batch_shape=batch_and_sequence_shape + ssc_input_shape, name="SSC Input")

    spl_input = tf.keras.layers.Concatenate()([vc_input, ssc_input])
    spl = tf.keras.layers.Dense(256)(spl_input)
    spl = tf.keras.layers.ReLU()(spl)

    ipl = tf.keras.layers.Dense(256)(spl)
    ipl = tf.keras.layers.ReLU()(ipl)

    ips_input = tf.keras.layers.Concatenate()([ipl, ssc_input])
    ips = tf.keras.layers.Dense(128)(ips_input)
    ips = tf.keras.layers.ReLU()(ips)

    return ips, ipl


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

    return mcc, lpfc


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
    pmc = tf.keras.layers.LSTM(512)(pmc)

    m1_input = tf.keras.layers.Concatenate([pmc, mcc_input, lpfc_input, ssc_input, ips_input])
    m1 = tf.keras.layers.Dense(256)(m1_input)
    m1 = tf.keras.layers.ReLU()(m1)

    return m1


def build_shadow_brain_base(env: gym.Env, distribution: BasePolicyDistribution, bs: int, model_type: str = "rnn",
                            blind: bool = False, sequence_length=1, **kwargs):
    """Build network for the shadow hand task, version 2."""
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32

    rnn_choice = {"rnn": tf.keras.layers.SimpleRNN,
                  "lstm": tf.keras.layers.LSTM,
                  "gru": tf.keras.layers.GRU}[
        model_type]

    # inputs
    proprio_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["proprioception"],),
                                name="proprioception")
    touch_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["somatosensation"],),
                              name="somatosensation")
    goal_in = tf.keras.Input(batch_shape=(bs, sequence_length, *state_dimensionality["goal"],), name="goal")
    input_list = [proprio_in, touch_in, goal_in]

    # concatenation of touch and proprioception
    proprio_touch = tf.keras.layers.Concatenate()([proprio_in, touch_in])
    proprio_touch_masked = tf.keras.layers.Masking(
        batch_input_shape=(bs, sequence_length,) + proprio_touch.shape)(proprio_touch)

    encoder_sub_model = _build_encoding_sub_model(
        proprio_touch_masked.shape[-1],
        bs * sequence_length,
        layer_sizes=[128, 64, 64, 32],
        name="proprio_touch_encoder")

    proprio_touch_latent = TD(encoder_sub_model, name="TD_policy")(proprio_touch_masked)
    proprio_touch_latent.set_shape([bs] + proprio_touch_latent.shape[1:])

    abstractions = [goal_in, proprio_touch_latent]

    # visual inputs
    if not blind:
        visual_in = tf.keras.Input(batch_shape=(bs, None, VISION_WH, VISION_WH, 3), name="vision")
        input_list.append(visual_in)
        visual_masked = tf.keras.layers.Masking(
            batch_input_shape=(bs, sequence_length,) + state_dimensionality["vision"])(visual_in)

        visual_encoder = _build_openai_small_encoder(shape=(VISION_WH, VISION_WH, 3), out_shape=7,
                                                     batch_size=bs * sequence_length)
        visual_latent = TD(visual_encoder)(visual_masked)
        visual_latent.set_shape([bs] + visual_latent.shape[1:])

        visual_plus_goal = tf.keras.layers.Concatenate()([visual_latent, goal_in])
        goal_vision = TD(tf.keras.layers.Dense(20))(visual_plus_goal)
        goal_vision = TD(tf.keras.layers.Activation("relu"))(goal_vision)
        goal_vision.set_shape([bs] + goal_vision.shape[1:])

        abstractions.append(goal_vision)

    # concatenation of goal and perception
    x = tf.keras.layers.Concatenate()(abstractions)

    # recurrent layer
    rnn_out, *_ = rnn_choice(hidden_dimensions,
                             stateful=True,
                             return_sequences=True,
                             batch_size=bs,
                             return_state=True,
                             name="policy_recurrent_layer")(x)

    # output heads
    policy_out = distribution.build_action_head(n_actions, rnn_out.shape[1:], bs)(rnn_out)
    value_out = tf.keras.layers.Dense(1, name="value")(rnn_out)

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

    env = make_env("ReachAbsoluteVisual-v0")
    _, _, joint = build_shadow_brain_base(env, BetaPolicyDistribution(env), bs=batch_size, blind=False,
                                          sequence_length=sequence_length)
    plot_model(joint, to_file=f"{joint.name}.png", expand_nested=True, show_shapes=True)
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD()

    joint({
        "vision": tf.random.normal((batch_size, sequence_length, VISION_WH, VISION_WH, 3)),
        "proprioception": tf.random.normal((batch_size, sequence_length, 48)),
        "somatosensation": tf.random.normal((batch_size, sequence_length, 92)),
        "goal": tf.random.normal((batch_size, sequence_length, 15)),
    })

    joint.summary()
