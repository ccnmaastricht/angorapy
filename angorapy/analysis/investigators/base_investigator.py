#!/usr/bin/env python
import copy
import os
from time import sleep
from typing import List, Union

import gym
import matplotlib.pyplot as plt
import tensorflow as tf

from angorapy.common.policies import BasePolicyDistribution
from angorapy.common.wrappers import BaseWrapper
from angorapy.agent.ppo_agent import PPOAgent
from angorapy.utilities.hooks import register_hook, clear_hooks
from angorapy.utilities.model_utils import is_recurrent_model, list_layer_names, get_layers_by_names, build_sub_model_to, \
    extract_layers, CONVOLUTION_BASE_CLASS, is_conv
from angorapy.utilities.util import add_state_dims, flatten, insert_unknown_shape_dimensions


class Investigator:
    """Interface for investigating a policy network.

    The class serves as a wrapper to use a collection of methods that can extract information about the network."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        """Build an investigator for a network using a distribution.

        Args:
            network (tf.keras.Model):               the policy network to investigate
            distribution (BasePolicyDistribution):     a distribution that the network predicts
        """
        self.network: tf.keras.Model = network
        self.distribution: BasePolicyDistribution = distribution
        self.is_recurrent = is_recurrent_model(self.network)

    @classmethod
    def from_agent(cls, agent: PPOAgent):
        """Instantiate an investigator from an agent object."""
        agent.policy, agent.value, agent.joint = agent.build_models(agent.joint.get_weights(),
                                                                    batch_size=1, sequence_length=1)
        return cls(agent.policy, agent.distribution)

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

    def get_layer_activations(self, layer_names: List[str], input_tensor=None):
        """Get activations of a layer. If no input tensor is given, a random tensor is used."""
        activations = {}

        def activation_hook(module, input, output):
            activations[module.name] = output

        register_hook(get_layers_by_names(self.network, layer_names), after_call=activation_hook)
        if input_tensor is None:
            input_tensor = tf.random.normal(insert_unknown_shape_dimensions(self.network.input_shape))

        output = self.network(input_tensor, training=False)
        activations["output"] = output

        for key in activations.keys():
            if isinstance(activations[key], list):
                activations[key] = activations[key][0]

        # release the hook to prevent infinite nesting
        clear_hooks(self.network)

        return activations

    def get_activations_over_episode(self, layer_names: Union[List[str], str], env: gym.Env, render: bool = False):
        """Run an episode using the network and get (serialization, activation, r) tuples for each timestep."""
        layer_names = layer_names if isinstance(layer_names, list) else [layer_names]

        states, activations, reward_trajectory, action_trajectory = [], [], [], []

        # make new model with multiple outputs
        polymodel = build_sub_model_to(self.network, layer_names, include_original=True)
        is_recurrent = is_recurrent_model(self.network)

        done = False
        state = env.reset()
        env.goal = 1
        env.render() if render else ""
        while not done:
            dual_out = flatten(polymodel.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))
            activation, probabilities = dual_out[:-len(self.network.output)], dual_out[-len(self.network.output):]

            states.append(state)
            activations.append(activation)

            action = self.distribution.act_deterministic(*probabilities)
            action_trajectory.append(action)
            observation, reward, done, info = env.step(action)
            observation, reward, done, info = self.preprocessor.modulate((parse_state(observation), reward, done, info),
                                                                      update=False)

            state = observation
            reward_trajectory.append(reward)

            env.render() if render else ""

        return [states, list(zip(*activations)), reward_trajectory, action_trajectory]

    def render_episode(self, env: gym.Env, substeps_per_step=1, act_confidently=True) -> None:
        """Render an episode in the given environment."""
        is_recurrent = is_recurrent_model(self.network)
        self.network.reset_states()

        done, step = False, 0

        state, _ = env.reset()
        cumulative_reward = 0
        while not done:
            step += 1

            prepared_state = state.with_leading_dims(time=is_recurrent).dict()
            probabilities = flatten(self.network(prepared_state, training=False))

            if act_confidently:
                action, _ = self.distribution.act_deterministic(*probabilities)
            else:
                action, _ = self.distribution.act(*probabilities)

            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += (reward if "original_reward" not in info else info["original_reward"])
            done = terminated or truncated

            state = observation

        print(f"Finished after {step} steps with a score of {round(cumulative_reward, 4)}. "
              f"{'Good Boy!' if env.spec.reward_threshold is not None and cumulative_reward > env.spec.reward_threshold else ''}")

    def render_episode_jupyter(self, env: BaseWrapper, substeps_per_step=1, act_confidently=True) -> None:
        """Render an episode in the given environment."""
        from IPython import display, core

        is_recurrent = is_recurrent_model(self.network)
        self.network.reset_states()

        done, step = False, 0

        state, _ = env.reset()
        cumulative_reward = 0
        img = plt.imshow(env.render())
        while not done:
            img.set_data(env.render())
            plt.axis("off")
            display.display(plt.gcf())
            display.clear_output(wait=True)

            step += 1

            prepared_state = state.with_leading_dims(time=is_recurrent).dict()
            probabilities = flatten(self.network(prepared_state, training=False))

            if act_confidently:
                action, _ = self.distribution.act_deterministic(*probabilities)
            else:
                action, _ = self.distribution.act(*probabilities)

            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += (reward if "original_reward" not in info else info["original_reward"])
            done = terminated or truncated

            state = observation

        return


if __name__ == "__main__":
    print("INVESTIGATING")
    os.chdir("../../../")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # agent_id = 1585500821  # cartpole-v1
    agent_id = 1583256614 # reach task
    agent_007 = PPOAgent.from_agent_state(agent_id, from_iteration="b")

    inv = Investigator.from_agent(agent_007)
    print(inv.list_layer_names())

    # inv.get_activations_over_episode("policy_recurrent_layer", agent_007.env)

    # inv.get_activations_over_episode("policy_recurrent_layer", agent_007.env)

    for i in range(100):
        inv.render_episode(agent_007.env)
