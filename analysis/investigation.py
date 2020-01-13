#!/usr/bin/env python
import os
from typing import List, Union

import gym
import tensorflow as tf

from agent.policies import _PolicyDistribution
from agent.ppo import PPOAgent
from utilities.model_management import is_recurrent_model, list_layer_names, get_layers_by_names, \
    build_sub_model_to
from utilities.util import parse_state, add_state_dims, flatten, \
    insert_unknown_shape_dimensions
from utilities.wrappers import RewardNormalizationWrapper, StateNormalizationWrapper, CombiWrapper, _Wrapper, \
    SkipWrapper


class Investigator:
    """Interface for investigating a policy network.

    The class serves as a wrapper to use a collection of methods that can extract information about the network."""

    def __init__(self, network: tf.keras.Model, distribution: _PolicyDistribution, preprocessor: _Wrapper = None):
        """Build an investigator for a network using a distribution.

        Args:
            network (tf.keras.Model):               the policy network to investigate
            distribution (_PolicyDistribution):     a distribution that the network predicts
        """
        self.network: tf.keras.Model = network
        self.distribution: _PolicyDistribution = distribution
        self.preprocessor: _Wrapper = preprocessor if preprocessor is not None else SkipWrapper()

    @staticmethod
    def from_agent(agent: PPOAgent):
        """Instantiate an investigator from an agent object."""
        return Investigator(agent.policy, agent.distribution, preprocessor=agent.preprocessor)

    def list_layer_names(self, only_para_layers=True) -> List[str]:
        """Get a list of unique string representations of all layers in the network."""
        return list_layer_names(self.network, only_para_layers=only_para_layers)

    def get_layers_by_names(self, layer_names: Union[List[str], str]):
        """Retrieve the layer object identified from the model by its unique string representation."""
        if not isinstance(layer_names, list):
            layer_names = [layer_names]
        return get_layers_by_names(self.network, layer_names=layer_names)

    def get_layer_weights(self, layer_name):
        """Get the weights of a layer identified by its unique name."""
        return self.get_layers_by_names(layer_name).get_weights()

    def get_weight_dict(self):
        """Get a dictionary mapping layer names to the weights of the layers."""
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

        states, activations, reward_trajectory, action_trajectory = [], [], [], []

        # make new model with multiple outputs
        polymodel = build_sub_model_to(self.network, layer_names, include_original=True)
        is_recurrent = is_recurrent_model(self.network)

        done = False
        state = env.reset()
        state = self.preprocessor.wrap_a_step((state, None, None, None))
        while not done:
            dual_out = flatten(polymodel.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))
            activation, probabilities = dual_out[:-len(self.network.output)], dual_out[-len(self.network.output):]

            states.append(state)
            activations.append(activation)
            env.render() if render else ""

            action, _ = self.distribution.act(*probabilities)
            action_trajectory.append(action)
            observation, reward, done, _ = env.step(action)
            observation, reward, done, _ = self.preprocessor.wrap_a_step((observation, reward, done, None),
                                                                         update=False)

            state = observation
            reward_trajectory.append(reward)

        return [states, list(zip(*activations)), reward_trajectory, action_trajectory]

    def render_episode(self, env: gym.Env):
        """Render an episode in the given environment."""
        is_recurrent = is_recurrent_model(self.network)

        done = False
        state = env.reset()
        state = self.preprocessor.wrap_a_step((state, None, None, None), update=False)
        cumulative_reward = 0
        while not done:
            env.render()
            probabilities = flatten(
                self.network.predict(add_state_dims(parse_state(state), dims=2 if is_recurrent else 1)))

            action, _ = self.distribution.act(*probabilities)
            observation, reward, done, _ = env.step(action)
            cumulative_reward += reward
            observation, reward, done, _ = self.preprocessor.wrap_a_step((observation, reward, done, None),
                                                                         update=False)

            state = observation

        print(
            f"Achieved a score of {cumulative_reward}. {'Good Boy!' if cumulative_reward > env.spec.reward_threshold else ''}")


if __name__ == "__main__":
    os.chdir("../")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    agent_007 = PPOAgent.from_agent_state(1578664065)
    agent_007.preprocessor = CombiWrapper(
        [StateNormalizationWrapper(agent_007.state_dim), RewardNormalizationWrapper()])
    inv = Investigator.from_agent(agent_007)

    inv.render_episode(agent_007.env)
