#!/usr/bin/env python
import os
from pprint import pprint
from typing import List

import tensorflow as tf
from gym.spaces import Discrete, Box

from agent.policy import act_discrete, act_continuous
from models import build_rnn_distinct_models
from utilities.util import extract_layers, is_recurrent_model, parse_state, add_state_dims


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

    def get_layer_weights(self, layer_name):
        return self.network.get_layer(layer_name).get_weights()

    def get_weight_dict(self):
        out = {}
        for layer_name in self.list_layer_names(only_para_layers=True):
            out[layer_name] = self.get_layer_weights(layer_name)

        return out

    def get_layer_activations(self, layer_name, input_tensor):
        layer = self.network.get_layer(layer_name)
        sub_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        return sub_model.predict(input_tensor)

    def get_activations_over_episode(self, layer_name, env, render: bool = False):
        states = []
        activations = []
        actions = []

        layer = self.network.get_layer(layer_name)
        dual_model = tf.keras.Model(inputs=self.network.input, outputs=[layer.output, self.network.output])

        if isinstance(env.action_space, Discrete):
            continuous_control = False
        elif isinstance(env.action_space, Box):
            continuous_control = True
        else:
            raise ValueError("Unknown action space.")

        is_recurrent = is_recurrent_model(self.network)
        policy_act = act_discrete if not continuous_control else act_continuous

        reward_trajectory = []
        env.render() if render else ""

        state = parse_state(env.reset())
        done = False
        while not done:
            activation, probabilities = dual_model.predict(add_state_dims(state, dims=2 if is_recurrent else 1))
            states.append(state)
            activations.append(activation)

            action, action_prob = policy_act(probabilities)
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            state = parse_state(observation)
            reward_trajectory.append(reward)

        return list(zip(states, activations, reward_trajectory, actions))

        # return reward_trajectory



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


