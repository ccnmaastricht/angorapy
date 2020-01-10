#!/usr/bin/env python
import os
from pprint import pprint
from typing import List, Union

import gym
import numpy as np
import tensorflow as tf

from agent.policy import _PolicyDistribution, GaussianPolicyDistribution
from models import build_ffn_models, get_model_builder, plot_model
from utilities.util import parse_state, add_state_dims, flatten, \
    insert_unknown_shape_dimensions
from utilities.model_management import is_recurrent_model, list_layer_names, extract_layers, get_layers_by_names, \
    build_sub_model_to


class Investigator:
    """Interface for investigating a policy network.

    The class serves as a wrapper to use a collection of methods that can extract information about the network."""

    def __init__(self, network: tf.keras.Model, distribution: _PolicyDistribution):
        """Build an investigator for a network using a distribution.

        Args:
            network (tf.keras.Model):               the policy network to investigate
            distribution (_PolicyDistribution):     a distribution that the network predicts
        """
        self.network: tf.keras.Model = network
        self.distribution: _PolicyDistribution = distribution

    def list_layer_names(self, only_para_layers=True) -> List[str]:
        """Get a list of unique string representations of all layers in the network."""
        return list_layer_names(self.network, only_para_layers=only_para_layers)

    def get_layers_by_names(self, layer_names: Union[List[str], str]):
        """Retrieve the layer object identified from the model by its unique string representation."""
        if not isinstance(layer_names, list):
            layer_names = [layer_names]
        return get_layers_by_names(self.network, layer_names=layer_names)

    def get_layer_weights(self, layer_name):
        return self.get_layers_by_names(layer_name).get_weights()

    def get_weight_dict(self):
        out = {}
        for layer_name in self.list_layer_names(only_para_layers=True):
            out[layer_name] = self.get_layer_weights(layer_name)

        return out

    def get_layer_activations(self, layer_name: str, input_tensor=None):
        """Get activations of a layer. If no input tensor is given, a random tensor is used."""

        # make a sub model to the requested layer
        sub_model = build_sub_model_to(self.network, [layer_name])

        if input_tensor is None:
            input_tensor = tf.random.normal(insert_unknown_shape_dimensions(sub_model.input_shape))

        return sub_model.predict(input_tensor)

    def get_activations_over_episode(self, layer_names: Union[List[str], str], env: gym.Env, render: bool = False):
        """Run an episode using the network and get (s, activation, r) tuples for each timestep."""
        layer_names = layer_names if isinstance(layer_names, list) else [layer_names]

        states = []
        activations = []

        dual_model = build_sub_model_to(self.network, layer_names, include_original=True)
        is_recurrent = is_recurrent_model(self.network)

        done = False
        reward_trajectory = []
        state = parse_state(env.reset())
        while not done:
            dual_out = flatten(dual_model.predict(add_state_dims(state, dims=2 if is_recurrent else 1)))
            activation, probabilities = dual_out[:len(layer_names)], dual_out[len(layer_names):]

            states.append(state)
            activations.append(activation)
            env.render() if render else ""

            action, _ = self.distribution.act(*probabilities)
            observation, reward, done, _ = env.step(action)
            state = parse_state(observation)
            reward_trajectory.append(reward)

        return list(zip(states, activations, reward_trajectory))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    environment = gym.make("LunarLanderContinuous-v2")
    network, _, _ = get_model_builder("rnn", False)(environment)
    inv = Investigator(network, GaussianPolicyDistribution())

    for ln in inv.list_layer_names():
        tuples = inv.get_activations_over_episode(ln, environment, True)
        pprint(tuples)
