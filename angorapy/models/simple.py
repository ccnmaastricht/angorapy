#!/usr/bin/env python
"""Collection of generic fully connected and recurrent policy networks."""

import os
from contextlib import suppress
from typing import Tuple

import gym
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
from tensorflow.python.keras.utils.vis_utils import plot_model

from angorapy.common.policies import BasePolicyDistribution, CategoricalPolicyDistribution, BetaPolicyDistribution, \
    MultiCategoricalPolicyDistribution, RBetaPolicyDistribution, GaussianPolicyDistribution
from angorapy.common.wrappers import BaseWrapper, make_env
from angorapy.models.components import _build_encoding_sub_model
from angorapy.utilities.util import env_extract_dims


def build_ffn_models(env: BaseWrapper,
                     distribution: BasePolicyDistribution,
                     shared: bool = False,
                     layer_sizes: Tuple = (64, 64)):
    """Build a simple fully connected feed-forward model model."""

    # preparation
    state_dimensionality, n_actions = env_extract_dims(env)

    input_list = []
    for sense_name in state_dimensionality.keys():
        if sense_name == "asymmetric":
            continue

        input_node = tf.keras.Input(shape=state_dimensionality[sense_name], name=sense_name)
        input_list.append(input_node)

    if len(input_list) > 1:
        inputs = tf.keras.layers.Concatenate(name="flat_inputs")(input_list)
    else:
        inputs = input_list[0]

    # policy network
    latent = _build_encoding_sub_model(inputs.shape[1:], None, layer_sizes=layer_sizes, name="policy_encoder")(inputs)
    out_policy = distribution.build_action_head(n_actions, (layer_sizes[-1],), None)(latent)

    policy = tf.keras.Model(inputs=input_list, outputs=out_policy, name="policy")

    # value network
    if "asymmetric" in state_dimensionality.keys():
        asymmetric_inputs = tf.keras.Input(shape=state_dimensionality["asymmetric"], name="asymmetric")
        inputs = tf.keras.layers.Concatenate(name="added_asymmetric_inputs")([inputs, asymmetric_inputs])
        input_list.append(asymmetric_inputs)
    if not shared:
        value_latent = _build_encoding_sub_model(inputs.shape[1:], None, layer_sizes=layer_sizes, name="value_encoder")(inputs)
        value_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(value_latent)
    else:
        value_out = tf.keras.layers.Dense(1, input_dim=layer_sizes[-1],
                                          kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(latent)

    value = tf.keras.Model(inputs=input_list, outputs=value_out, name="value")

    return policy, value, tf.keras.Model(inputs=input_list, outputs=[out_policy, value_out], name="policy_value")


def build_rnn_models(env: BaseWrapper,
                     distribution: BasePolicyDistribution,
                     shared: bool = False,
                     bs: int = 1,
                     model_type: str = "gru",
                     layer_sizes: Tuple = (64, 64),
                     sequence_length=1):
    """Build simple policy and value models having a recurrent layer before their heads.

    Args:
        sequence_length:
    """
    # TODO: remove current workaround: solve issue with stateful masked RNNs by fixing the models sequence length

    state_dimensionality, n_actions = env_extract_dims(env)

    input_list = []
    for sense_name in state_dimensionality.keys():
        if sense_name == "asymmetric":
            continue

        input_node = tf.keras.Input(batch_shape=(bs, sequence_length,) + state_dimensionality[sense_name], name=sense_name)
        input_list.append(input_node)

    if len(input_list) > 1:
        inputs = tf.keras.layers.Concatenate(name="flat_inputs")(input_list)
    else:
        inputs = input_list[0]

    if "asymmetric" in state_dimensionality.keys():
        asymmetric_inputs = tf.keras.Input(batch_shape=(bs, sequence_length,) + state_dimensionality["asymmetric"], name="asymmetric")
        value_inputs = tf.keras.layers.Concatenate(name="added_asymmetric_inputs")([inputs, asymmetric_inputs])
        input_list.append(asymmetric_inputs)
    else:
        value_inputs = inputs

    masked = tf.keras.layers.Masking(batch_input_shape=(bs, sequence_length,) + (inputs.shape[-1], ))(inputs)

    # policy network; stateful, so batch size needs to be known
    encoder_sub_model = _build_encoding_sub_model(
        inputs.shape[-1],
        # since we are distributing the encoder over all timesteps, we need to blow up the bs here
        bs * sequence_length,
        layer_sizes=layer_sizes[:-1],
        name="policy_encoder")

    x = TimeDistributed(encoder_sub_model, name="TD_policy")(masked)
    x.set_shape([bs] + x.shape[1:])

    rnn_choice = {"rnn": tf.keras.layers.SimpleRNN,
                  "lstm": tf.keras.layers.LSTM,
                  "gru": tf.keras.layers.GRU}[model_type]

    x, *_ = rnn_choice(layer_sizes[-1],
                       stateful=True,
                       return_sequences=True,
                       return_state=True,
                       batch_size=bs,
                       name="policy_recurrent_layer")(x)

    out_policy = distribution.build_action_head(n_actions, x.shape[1:], bs)(x)
    policy = tf.keras.Model(inputs=input_list, outputs=out_policy, name="simple_rnn_policy")

    # value network
    if not shared:
        value_inputs_masked = tf.keras.layers.Masking(batch_input_shape=(bs, sequence_length,) + (value_inputs.shape[-1],))(value_inputs)
        x = TimeDistributed(
            _build_encoding_sub_model(value_inputs_masked.shape[-1],
                                      bs * sequence_length,
                                      layer_sizes=layer_sizes[:-1],
                                      name="value_encoder"),
            name="TD_value")(value_inputs_masked)
        x.set_shape([bs] + x.shape[1:])

        x, *_ = rnn_choice(layer_sizes[-1], stateful=True, return_sequences=True, return_state=True, batch_size=bs,
                           name="value_recurrent_layer")(x)
        out_value = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0), name="value_out")(x)
    else:
        if "asymmetric" in state_dimensionality.keys():
            x = tf.keras.layers.Concatenate()([x, asymmetric_inputs])

        out_value = tf.keras.layers.Dense(1, input_dim=x.shape[1:],
                                          kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0), name="value_out")(x)

    value = tf.keras.Model(inputs=input_list, outputs=out_value, name="simple_rnn_value")

    return policy, value, tf.keras.Model(inputs=input_list, outputs=[out_policy, out_value], name="simple_rnn")


def build_simple_models(env: BaseWrapper,
                        distribution: BasePolicyDistribution,
                        shared: bool = False,
                        bs: int = 1,
                        sequence_length: int = None,
                        model_type: str = "gru",
                        **kwargs):
    """Build simple networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for ffn and rnns
    if model_type == "ffn":
        return build_ffn_models(env, distribution, shared, layer_sizes=(64, 64))
    else:
        return build_rnn_models(env, distribution, shared, bs=bs, sequence_length=sequence_length, model_type=model_type,
                                layer_sizes=(64, 64))


def build_deeper_models(env: BaseWrapper,
                        distribution: BasePolicyDistribution,
                        shared: bool = False,
                        bs: int = 1,
                        sequence_length: int = 1,
                        model_type: str = "gru",
                        **kwargs):
    """Build deeper simple networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for ffn and rnns
    if model_type == "ffn":
        return build_ffn_models(env, distribution, shared, layer_sizes=(64, 64, 64, 32))
    else:
        return build_rnn_models(env, distribution, shared, bs=bs, sequence_length=sequence_length,
                                model_type=model_type, layer_sizes=(64, 64, 64, 32, 32))


def build_wider_models(env: BaseWrapper,
                       distribution: BasePolicyDistribution,
                       shared: bool = False,
                       bs: int = 1,
                       sequence_length: int = 1,
                       model_type: str = "gru",
                       **kwargs):
    """Build deeper simple networks (policy, value, joint) for given parameter settings."""

    # this function is just a wrapper routing the requests for ffn and rnns
    if model_type == "ffn":
        return build_ffn_models(env, distribution, shared, layer_sizes=(1024, 512))
    else:
        return build_rnn_models(env, distribution, shared, bs=bs, sequence_length=sequence_length,
                                model_type=model_type, layer_sizes=(1024, 512))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    cont_env = make_env("HumanoidManipulateBlockDiscreteAsynchronous-v0")
    discrete_env = gym.make("LunarLander-v2")
    multi_discrete_env = gym.make("ManipulateBlockDiscreteRelative-v0")

    # model = build_simple_models(cont_env, RBetaPolicyDistribution(cont_env), False, 1, 1, "gru")
    # print(f"Simple model has {model[0].count_params()} parameters.")
    # plot_model(model[2], show_shapes=True, to_file="simple.png", expand_nested=True)
    #
    # model = build_simple_models(discrete_env, CategoricalPolicyDistribution(discrete_env), False, 1, 1, "gru")
    # print(f"Simple model has {model[0].count_params()} parameters.")
    # plot_model(model[2], show_shapes=True, to_file="model_graph_simple_discrete.png", expand_nested=True)
    #
    # model = build_simple_models(multi_discrete_env, MultiCategoricalPolicyDistribution(multi_discrete_env), False, 1, 1, "gru")
    # print(f"Simple model has {model[0].count_params()} parameters.")
    # plot_model(model[2], show_shapes=True, to_file="model_graph_simple_multidiscrete.png", expand_nested=True)
    #
    # model = build_deeper_models(cont_env, BetaPolicyDistribution(cont_env), False, 1, 1, "gru")
    # print(f"Deeper model has {model[0].count_params()} parameters.")
    # plot_model(model[2], show_shapes=True, to_file="model_graph_deeper.png", expand_nested=True)

    model = build_wider_models(cont_env, MultiCategoricalPolicyDistribution(cont_env), False, 1, 1, "gru")
    print(f"Wider model without weight sharing has {model[2].count_params()} parameters.")
    plot_model(model[2], show_shapes=True, to_file="model_graph_wider_unshared.png", expand_nested=True)

    model = build_wider_models(cont_env, MultiCategoricalPolicyDistribution(cont_env), False, 1, 1, "gru")
    print(f"Wider model without weight sharing has {model[2].count_params()} parameters.")
    plot_model(model[2], show_shapes=True, to_file="model_graph_wider_unshared_asymmetric.png", expand_nested=True)

    model = build_wider_models(cont_env, MultiCategoricalPolicyDistribution(cont_env), True, 1, 1, "gru")
    print(f"Wider model with weight sharing has {model[2].count_params()} parameters.")
    plot_model(model[2], show_shapes=True, to_file="model_graph_wider_shared.png", expand_nested=True)
