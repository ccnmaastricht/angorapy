#!/usr/bin/env python
import os
from pprint import pprint
from typing import List

import gym
import numpy as np
import tensorflow as tf

from agent.policy import _PolicyDistribution, GaussianPolicyDistribution
from models import build_ffn_models, get_model_builder, plot_model
from utilities.util import extract_layers, is_recurrent_model, parse_state, add_state_dims, flatten, \
    insert_unknown_shape_dimensions, list_layer_names


class Investigator:

    def __init__(self, network, distribution: _PolicyDistribution):
        self.network = network
        self.distribution = distribution

    def list_layer_names(self, only_para_layers=True) -> List[str]:
        """Get a list of unique string representations of all layers in the network."""
        return list_layer_names(self.network, only_para_layers=only_para_layers)

    def get_layer_by_name(self, layer_name):
        """Retrieve the layer object identified from the model by its unique string representation."""
        return extract_layers(self.network)[self.list_layer_names(only_para_layers=False).index(layer_name)]

    def get_layer_weights(self, layer_name):
        return self.get_layer_by_name(layer_name).get_weights()

    def get_weight_dict(self):
        out = {}
        for layer_name in self.list_layer_names(only_para_layers=True):
            out[layer_name] = self.get_layer_weights(layer_name)

        return out

    def get_layer_activations(self, layer_name, input_tensor=None):
        """Get activations of a layer. If no input tensor is given, a random tensor is used."""

        # make a sub model to the requested layer
        layer = self.get_layer_by_name(layer_name)
        print(layer)
        success = False
        layer_input_id = 0
        while not success:
            success = True
            try:
                sub_model = tf.keras.Model(inputs=[self.network.input], outputs=[layer.get_output_at(layer_input_id)])
            except ValueError as ve:
                if len(ve.args) > 0 and ve.args[0].split(" ")[0] == "Graph":
                    print(ve.args)
                    layer_input_id += 1
                    success = False
                else:
                    raise ValueError(f"Cannot use layer {layer.name}. Error: {ve.args}")

        if input_tensor is None:
            input_tensor = tf.random.normal(insert_unknown_shape_dimensions(sub_model.input_shape))

        return sub_model.predict(input_tensor)

    def get_activations_over_episode(self, layer_name: List[str], env, render: bool = False):
        states = []
        activations = []

        layer = self.get_layer_by_name(layer_name)
        dual_model = tf.keras.Model(inputs=self.network.input, outputs=[layer.output, self.network.output])

        is_recurrent = is_recurrent_model(self.network)

        done = False
        reward_trajectory = []
        state = parse_state(env.reset())
        while not done:
            dual_out = flatten(dual_model.predict(add_state_dims(state, dims=2 if is_recurrent else 1)))
            activation, probabilities = dual_out[0], dual_out[1:]

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

    env = gym.make("LunarLanderContinuous-v2")
    network, _, _ = get_model_builder("rnn", False)(env)
    network.summary()
    inv = Investigator(network, GaussianPolicyDistribution())


    print(inv.list_layer_names())
    plot_model(network, "lalala.png")

    print(f"\nInput:")
    input_tensor = tf.random.normal(insert_unknown_shape_dimensions(network.input_shape))
    print(input_tensor)
    for layer_name in inv.list_layer_names():
        print(f"\nActivations of {layer_name}:")
        activation_rec = inv.get_layer_activations(layer_name, input_tensor=input_tensor)
        print(activation_rec)
        # print(f"Weights of {layer_name}:")
        # print(inv.get_layer_weights(layer_name))

    # tuples = inv.get_activations_over_episode("dense_1", env, True)
    # print(len(tuples))
    # pprint(list(zip(*tuples))[1])
    #
    # # tsne_results = sklm.TSNE.fit_transform(np.array(tuples[0]))
    # state_data = np.empty((len(np.array(tuples)[:, 0]), 8))
    #
    # for l in np.array(tuples)[:, 0]:
    #     state_data += l
