#!/usr/bin/env python
import os
from pprint import pprint
from typing import List

import numpy as np

import gym
import tensorflow as tf
from gym.spaces import Discrete, Box
import gym
from agent.ppo import PPOAgent
from environments import *

from agent.policy import act_discrete, act_continuous

from models import build_rnn_models, build_ffn_models
from utilities.util import extract_layers, is_recurrent_model, parse_state, add_state_dims, flatten, \
    insert_unknown_shape_dimensions


class Investigator:

    def __init__(self, network):
        self.network = network

    def list_layer_names(self, only_para_layers=False) -> List[str]:
        """Get a list of unique string representations of all layers in the network."""
        if only_para_layers:
            return [layer.name for layer in extract_layers(self.network) if
                    not isinstance(layer, tf.keras.layers.Activation)]
        else:
            return [layer.name for layer in extract_layers(self.network)]

    def get_layer_by_name(self, layer_name):
        """Retrieve the layer object identified from the model by its unique string representation."""
        return extract_layers(self.network)[self.list_layer_names().index(layer_name)]

    def get_layer_weights(self, layer_name):
        return self.get_layer_by_name(layer_name).get_weights()

    def get_weight_dict(self):
        out = {}
        for layer_name in self.list_layer_names(only_para_layers=True):
            out[layer_name] = self.get_layer_weights(layer_name)

        return out

    def get_layer_activations(self, layer_name, input_tensor=None):
        """Get activations of a layer. If no input tensor is given, a random tensor is used."""
        layer = self.get_layer_by_name(layer_name)
        sub_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        if input_tensor is None:
            input_tensor = tf.random.normal(insert_unknown_shape_dimensions(sub_model.input_shape))

        return sub_model.predict(input_tensor)

    def get_activations_over_episode(self, layer_name, previous_layer_name, env, render: bool = False):
        states = []
        activations = []
        actions = []
        previous_activations = []

        layer = self.get_layer_by_name(layer_name)
        previous_layer = self.get_layer_by_name(previous_layer_name)
        dual_model = tf.keras.Model(inputs=self.network.input, outputs=[layer.output,
                                                                        previous_layer.output,
                                                                        self.network.output])

        if isinstance(env.action_space, Discrete):
            continuous_control = False
        elif isinstance(env.action_space, Box):
            continuous_control = True
        else:
            raise ValueError("Unknown action space.")

        is_recurrent = is_recurrent_model(self.network)

        reward_trajectory = []
        env.render() if render else ""

        state = parse_state(env.reset())
        done = False
        while not done:
            dual_out = flatten(dual_model.predict(add_state_dims(state, dims=2 if is_recurrent else 1)))
            activation, previous_activation, probabilities = dual_out[0], dual_out[1], dual_out[2:]

            states.append(state)
            # action, action_probability = act_continuous(*a_distr) if is_continuous else act_discrete(*a_distr)
            activations.append(activation)
            previous_activations.append(previous_activation)

            action, _ = act_continuous(*probabilities) if continuous_control else act_discrete(*probabilities)
            actions.append(action)
            observation, reward, done, _ = env.step(action)

            state = parse_state(observation)
            reward_trajectory.append(reward)

        return list(zip(states, activations, previous_activations, reward_trajectory, actions))

        # return reward_trajectory



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env_name = "LunarLanderContinuous-v2"
    agent_id = 1576849128
    env = gym.make(env_name)
    new_agent = PPOAgent.from_agent_state(agent_id)

    investi = Investigator(new_agent.policy)
    layer_names = investi.list_layer_names()
    print(investi.list_layer_names())

    activations = investi.get_layer_activations("policy_recurrent_layer")

    # env_name = "CartPole-v1"
    # agent_id: int = 1575394142
    # env = gym.make(env_name)
    # new_agent = PPOAgent.from_agent_state(agent_id)

    # investi = Investigator(new_agent.policy)

    # print(investi.list_layer_names())

    # weights = investi.get_layer_weights("lstm")
    # print(weights)


    # activations = investi.get_layer_activations("lstm", )

    # tuples = investi.get_activations_over_episode("lstm", env, True)

    # env = gym.make("LunarLander-v2")

    #env = gym.make("LunarLanderContinuous-v2")

    #network, _, _ = build_ffn_models(env)

    #inv = Investigator(network)

    #print(inv.list_layer_names())

    #activation_rec = inv.get_layer_activations("dense_1")
    #print(activation_rec)

    #tuples = inv.get_activations_over_episode("dense_1", env, True)

    #print(len(tuples))
    #pprint(list(zip(*tuples))[1])

    # tsne_results = sklm.TSNE.fit_transform(np.array(tuples[0]))
    #state_data = np.empty((len(np.array(tuples)[:, 0]), 8))

    #for l in np.array(tuples)[:, 0]:
    #    state_data += l


