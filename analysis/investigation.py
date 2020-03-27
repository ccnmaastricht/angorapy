#!/usr/bin/env python
import os
from typing import List, Union

import gym
import tensorflow as tf

from agent.policies import BasePolicyDistribution
from agent.ppo import PPOAgent
from utilities.model_utils import is_recurrent_model, list_layer_names, get_layers_by_names, build_sub_model_to, \
    extract_layers, CONVOLUTION_BASE_CLASS, is_conv
from utilities.util import parse_state, add_state_dims, flatten, insert_unknown_shape_dimensions
from utilities.wrappers import BaseWrapper, SkipWrapper


class Investigator:
    """Interface for investigating a policy network.

    The class serves as a wrapper to use a collection of methods that can extract information about the network."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution, preprocessor: BaseWrapper = None):
        """Build an investigator for a network using a distribution.

        Args:
            network (tf.keras.Model):               the policy network to investigate
            distribution (BasePolicyDistribution):     a distribution that the network predicts
        """
        self.network: tf.keras.Model = network
        self.distribution: BasePolicyDistribution = distribution
        self.preprocessor: BaseWrapper = preprocessor if preprocessor is not None else SkipWrapper()

    @staticmethod
    def from_agent(agent: PPOAgent):
        """Instantiate an investigator from an agent object."""
        return Investigator(agent.policy, agent.distribution, preprocessor=agent.preprocessor)

    def list_layer_names(self, only_para_layers=True) -> List[str]:
        """Get a list of unique string representations of all layers in the network."""
        return list_layer_names(self.network, only_para_layers=only_para_layers)

    def list_convolutional_layer_names(self) -> List[str]:
        """Get a list of unique string representations of convolutional layers in the network."""
        return [layer.name for layer in extract_layers(self.network) if is_conv(layer)]

    def list_non_convolutional_layer_names(self) -> List[str]:
        """Get a list of unique string representations of non-convolutional layers in the network."""
        return [layer.name for layer in extract_layers(self.network) if
                not isinstance(layer, CONVOLUTION_BASE_CLASS) and not isinstance(layer, tf.keras.layers.Activation)]

    def get_layers_by_names(self, layer_names: Union[List[str], str]):
        """Retrieve the layer object identified from the model by its unique string representation."""
        if not isinstance(layer_names, list):
            layer_names = [layer_names]
        return get_layers_by_names(self.network, layer_names=layer_names)

    def get_layer_by_name(self, layer_name):
        """Retrieve the layer object identified from the model by its unique string representation."""
        return self.get_layers_by_names(layer_name)[0]

    def get_layer_weights(self, layer_name):
        """Get the weights of a layer identified by its unique name."""
        return self.get_layers_by_names(layer_name)[0].get_weights()

    def get_weight_dict(self):
        """Get a dictionary mapping layer names to the weights of the layers."""
        out = {}
        for layer_name in self.list_layer_names(only_para_layers=True):
            out[layer_name] = self.get_layer_weights(layer_name)

        return out

    def plot_model(self):
        """Plot the network graph into a file."""
        tf.keras.utils.plot_model(self.network, show_shapes=True)

    def dissect_recurrent_layer_weights(self, layer_name):
        """Return a recurrent cells weights and biases in a named dictionary."""
        layer = self.get_layers_by_names(layer_name)[0]

        if not is_recurrent_model(layer):
            raise ValueError("Cannot dissect non-recurrent layer...")

        units = layer.units

        # stolen from https://stackoverflow.com/a/51484524/5407682
        W = layer.get_weights()[0]
        U = layer.get_weights()[1]
        b = layer.get_weights()[2]

        if isinstance(layer, tf.keras.layers.SimpleRNN):
            return dict(
                W=W,
                U=U,
                b=b
            )
        elif isinstance(layer, tf.keras.layers.GRU):
            return dict(
                W_z=W[:, :units],
                W_r=W[:, units: units * 2],
                W_c=W[:, units * 2:],

                U_z=U[:, :units],
                U_r=U[:, units: units * 2],
                U_c=U[:, units * 2:],

                b_z=b[:units],
                b_r=b[units: units * 2],
                b_c=b[units * 2:],
            )
        elif isinstance(layer, tf.keras.layers.LSTM):
            return dict(
                W_i=W[:, :units],
                W_f=W[:, units: units * 2],
                W_c=W[:, units * 2: units * 3],
                W_o=W[:, units * 3:],

                U_i=U[:, :units],
                U_f=U[:, units: units * 2],
                U_c=U[:, units * 2: units * 3],
                U_o=U[:, units * 3:],

                b_i=b[:units],
                b_f=b[units: units * 2],
                b_c=b[units * 2: units * 3],
                b_o=b[units * 3:],
            )
        else:
            raise ValueError("Recurrent layer type not understood. Is it custom?")

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

        states, activations, reward_trajectory, action_trajectory = [], [], [], []

        # make new model with multiple outputs
        polymodel = build_sub_model_to(self.network, layer_names, include_original=True)
        is_recurrent = is_recurrent_model(self.network)

        done = False
        state = env.reset()
        state = self.preprocessor.modulate((parse_state(state), None, None, None))[0]
        while not done:
            dual_out = flatten(polymodel.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))
            activation, probabilities = dual_out[:-len(self.network.output)], dual_out[-len(self.network.output):]

            states.append(state)
            activations.append(activation)
            env.render() if render else ""

            action, _ = self.distribution.act(*probabilities)
            action_trajectory.append(action)
            observation, reward, done, _ = env.step(action)
            observation, reward, done, _ = self.preprocessor.modulate((parse_state(observation), reward, done, None),
                                                                      update=False)

            state = observation
            reward_trajectory.append(reward)

        return [states, list(zip(*activations)), reward_trajectory, action_trajectory]

    def render_episode(self, env: gym.Env, to_gif: bool = False) -> None:
        """Render an episode in the given environment."""
        is_recurrent = is_recurrent_model(self.network)
        self.network.reset_states()

        done = False
        state = self.preprocessor.modulate((parse_state(env.reset()), None, None, None), update=False)[0]
        cumulative_reward = 0
        while not done:
            env.render() if not to_gif else env.render(mode="rgb_array")
            probabilities = flatten(
                self.network.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))

            action, _ = self.distribution.act(*probabilities)
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            observation, reward, done, _ = self.preprocessor.modulate((parse_state(observation), reward, done, None),
                                                                      update=False)

            state = observation

        print(f"Achieved a score of {cumulative_reward}. "
              f"{'Good Boy!' if env.spec.reward_threshold is not None and cumulative_reward > env.spec.reward_threshold else ''}")


if __name__ == "__main__":
    print("INVESTIGATING")
    os.chdir("../")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    agent_007 = PPOAgent.from_agent_state(1583256614, from_iteration="b")
    inv = Investigator.from_agent(agent_007)
    print(inv.list_layer_names())

    #inv.get_activations_over_episode("policy_recurrent_layer", agent_007.env)

    #inv.get_activations_over_episode("policy_recurrent_layer", agent_007.env)

    for i in range(100):
        inv.render_episode(agent_007.env, to_gif=False)
