import os
import gym
import numpy as np
from time import sleep
import sys
sys.path.append("/Users/Raphael/dexterous-robot-hand/rnn_dynamical_systems")

from rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points
from agent.ppo import PPOAgent
from analysis.investigation import Investigator
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper
from utilities.util import parse_state, add_state_dims, flatten, insert_unknown_shape_dimensions
from utilities.model_utils import build_sub_model_from, build_sub_model_to


class Chiefinvestigator(Investigator):

    def __init__(self, agent_id: int, enforce_env_name: str = None):
        """Chiefinvestigator can assign investigator to inspect the model and produce high-level analysis.

        Args:
            agent_id: ID of agent that will be analyzed.
            env_name: Name of the gym environment that the agent was trained in. Default is set to CartPole-v1
        """

        self.agent = PPOAgent.from_agent_state(agent_id, from_iteration='best')
        super().__init__(self.agent.policy, self.agent.distribution, self.agent.preprocessor)
        self.env = self.agent.env
        if enforce_env_name is not None:
            print(f"Enforcing environment {enforce_env_name} over agents original environment. If you want to use"
                  f"the same environment as the original agent anyways, there is no need to specify it in the"
                  f"constructor!")
            self.env = gym.make(enforce_env_name)
        self.agent.preprocessor = CombiWrapper([StateNormalizationWrapper(self.agent.state_dim),
                                                RewardNormalizationWrapper()])  # dirty fix, TODO remove soon
        self.weights = self.get_layer_weights('policy_recurrent_layer')
        self.n_hidden = self.weights[1].shape[0]
        self._get_rnn_type()
        self.sub_model_from = build_sub_model_from(self.network, "policy_recurrent_layer")

    def _get_rnn_type(self):
        if self.weights[1].shape[1] / self.weights[1].shape[0] == 1:
            self.rnn_type = 'vanilla'
        elif self.weights[1].shape[1] / self.weights[1].shape[0] == 3:
            self.rnn_type = 'gru'
        elif self.weights[1].shape[1] / self.weights[1].shape[0] == 4:
            self.rnn_type = 'lstm'

    def get_layer_names(self):
        return self.list_layer_names()

    def parse_data(self, layer_name: str, previous_layer_name: str):
        """Get state, activation, action, reward over one episode. Parse data to output.

        Args:
            layer_name: Name of layer to be analysed

        Returns:
            activation_data, action_data, state_data, all_rewards
        """
        states, activations, rewards, actions = self.get_activations_over_episode(
            [layer_name, previous_layer_name],
            self.env, False)

        # merge activations per layer into matrices where first dimension are timesteps
        activations = list(map(np.array, activations))

        return states, activations, rewards, actions

    def get_data_over_episodes(self, n_episodes: int, layer_name: str, previous_layer_name: str):

        activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes = [], [], []
        for i in range(n_episodes):
            states, activations, rewards, actions = self.parse_data(layer_name, previous_layer_name)
            inputs = np.reshape(activations[2], (activations[2].shape[0], self.n_hidden))
            activations = np.reshape(activations[1], (activations[1].shape[0], self.n_hidden))
            activations_over_all_episodes.append(activations)
            inputs_over_all_episodes.append(inputs)
            actions = np.vstack(actions)
            actions_over_all_episodes.append(actions)

        activations_over_all_episodes, inputs_over_all_episodes = np.vstack(activations_over_all_episodes), \
                                                                  np.vstack(inputs_over_all_episodes)
        actions_over_all_episodes = np.concatenate(actions_over_all_episodes, axis=0)

        return activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes

    def get_data_over_single_run(self, layer_name: str, previous_layer_name: str):

        states, activations, rewards, actions = self.parse_data(layer_name, previous_layer_name)
        inputs = np.reshape(activations[2], (activations[2].shape[0], self.n_hidden))
        activations = np.reshape(activations[1], (activations[1].shape[0], self.n_hidden))
        actions = np.vstack(actions)

        return activations, inputs, actions

    def render_fixed_points(self, activations):

        self.env.reset()
        for i in range(100):
            self.env.render()
            activation = activations[0, i, :].reshape(1, 1, self.n_hidden)
            probabilities = flatten(self.sub_model_from.predict(activation))

            action, _ = self.distribution.act(*probabilities)
            observation, reward, done, info = self.env.step(action)


if __name__ == "__main__":
    os.chdir("../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1583404415  # 1583180664 lunarlandercont
    agent_id = 1585500821 # cartpole-v1
    # agent_id = 1583256614 # reach task
    chiefinvesti = Chiefinvestigator(agent_id)

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)

    activations_over_all_episodes, inputs_over_all_episodes, \
    actions_over_all_episodes = chiefinvesti.get_data_over_episodes(30,
                                                                    "policy_recurrent_layer",
                                                                    layer_names[1])

    # employ fixedpointfinder
    adamfpf = Adamfixedpointfinder(chiefinvesti.weights, chiefinvesti.rnn_type,
                                   q_threshold=1e-07,
                                   epsilon=0.01,
                                   alr_decayr=1e-04,
                                   max_iters=5000)
    states, sampled_inputs = adamfpf.sample_inputs_and_states(activations_over_all_episodes,
                                                              inputs_over_all_episodes,
                                                              2000, 0.2)
    sampled_inputs = np.zeros((states.shape[0], chiefinvesti.n_hidden))
    fps = adamfpf.find_fixed_points(states, sampled_inputs)
    plot_fixed_points(activations_over_all_episodes, fps, 4000, 1)
    # render fixedpoints
    #for fp in fps:
    #    repeated_fps = np.repeat(fp['x'].reshape(1, 1, chiefinvesti.n_hidden), 100, axis=1)
    #    print("RENDERING FIXED POINT")
    #    chiefinvesti.render_fixed_points(repeated_fps)
     #   sleep(5)



