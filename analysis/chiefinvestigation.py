import os
import gym
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition as skld
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

from agent.ppo import PPOAgent
from analysis.investigation import Investigator
# from utilities.util import insert_unknown_shape_dimensions
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper
from fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from mayavi import mlab
from fixedpointfinder.plot_utils import plot_fixed_points, plot_velocities
from analysis.rsa.rsa import RSA
from sklearn.decomposition import pca

class Chiefinvestigator:

    def __init__(self, agent_id: int, enforce_env_name: str = None):
        """Chiefinvestigator can assign investigator to inspect the model and produce high-level analysis.

        Args:
            agent_id: ID of agent that will be analyzed.
            env_name: Name of the gym environment that the agent was trained in. Default is set to CartPole-v1
        """
        self.agent = PPOAgent.from_agent_state(agent_id, from_iteration='best')
        self.env = self.agent.env
        if enforce_env_name is not None:
            print(f"Enforcing environment {enforce_env_name} over agents original environment. If you want to use"
                  f"the same environment as the original agent anyways, there is no need to specify it in the"
                  f"constructor!")
            self.env = gym.make(enforce_env_name)
        self.agent.preprocessor = CombiWrapper([StateNormalizationWrapper(self.agent.state_dim),
                                                RewardNormalizationWrapper()])  # dirty fix, TODO remove soon
        self.slave_investigator = Investigator.from_agent(self.agent)

    def get_layer_names(self):
        return self.slave_investigator.list_layer_names()

    def parse_data(self, layer_name: str, previous_layer_name: str):
        """Get state, activation, action, reward over one episode. Parse data to output.

        Args:
            layer_name: Name of layer to be analysed

        Returns:
            activation_data, action_data, state_data, all_rewards
        """
        states, activations, rewards, actions = self.slave_investigator.get_activations_over_episode(
            [layer_name, previous_layer_name],
            self.env, False
        )

        # merge activations per layer into matrices where first dimension are timesteps
        activations = list(map(np.array, activations))

        # activations = np.reshape(activations, (activations.shape[0], 64))
        # previous_activation = np.reshape(previous_activation, (previous_activation.shape[0], 64))

        return states, activations, rewards, actions

    def get_data_over_episodes(self, n_episodes: int, layer_name: str, previous_layer_name: str):

        activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes = [], [], []
        for i in range(n_episodes):
            states, activations, rewards, actions = chiefinvesti.parse_data(layer_name, previous_layer_name)
            inputs = np.reshape(activations[2], (activations[2].shape[0], 32))
            activations = np.reshape(activations[1], (activations[1].shape[0], 32))
            activations_over_all_episodes.append(activations)
            inputs_over_all_episodes.append(inputs)
            actions = np.vstack(actions)
            actions_over_all_episodes.append(actions)

        activations_over_all_episodes, inputs_over_all_episodes = np.vstack(activations_over_all_episodes), \
                                                                  np.vstack(inputs_over_all_episodes)
        actions_over_all_episodes = np.concatenate(actions_over_all_episodes, axis=0)

        return activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes

# TODO: build neural network to predict grasp -> we trained a simple prediction model fp(z) containing one hidden
# layer with 64 units and ReLU activation, followed by a sigmoid output.
# TODO: take reaching task as state not an action as the hand stays rather the same.
# TODO: classify reaching tasks against each other

# TODO: sequence analysis ideas -> sequence pattern and so forth:
# one possibility could be sequence alignment, time series analysis (datacamp), rsa

if __name__ == "__main__":
    os.chdir("../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1583404415 # 1583180664 # lunarlandercont
    agent_id = 1583256614 # reach task
    chiefinvesti = Chiefinvestigator(agent_id)

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)


    weights = chiefinvesti.slave_investigator.get_layer_weights('policy_recurrent_layer')
    activations_over_all_episodes, inputs_over_all_episodes, \
    activations_over_all_episodes = chiefinvesti.get_data_over_episodes(n_episodes=20,
                                                                        "policy_recurrent_layer",layer_names[1])
    findfixedpoints = True
    if findfixedpoints:
        adamfpf = Adamfixedpointfinder(weights, 'gru',
                                       q_threshold=1e-04,
                                       epsilon=0.001,
                                       alr_decayr=1e-04,
                                       max_iters=5000)
        statis, sampled_inputs = adamfpf.sample_inputs_and_states(activations_over_all_episodes,
                                                                  inputs_over_all_episodes,
                                                                  100, 0.2)
        sampled_inputs = np.zeros((statis.shape[0], 32))
        fps = adamfpf.find_fixed_points(statis, sampled_inputs)
        #vel = adamfpf.compute_velocities(activationss, inputss)
        plot_fixed_points(activations_over_all_episodes, fps, 4000, 1)
        # plot_velocities(activations_over_all_episodes, actions_over_all_episodes[:, 0], 7000)

        left_engine = actions_over_all_episodes
        engine_off = left_engine[:, 1] > -0.5
        left_engine[engine_off, 1] = 0
        # plot_velocities(activations_over_all_episodes, left_engine[:, 1], 3000)

        rsa = RSA(actions)
        rdm_time = rsa.compare_cross_timeintervals(1)
        # rdm_neurons = rsa.compare_cross_neurons()
        plt.imshow(rdm_time)
        plt.colorbar()
        plt.show()

    # pca
    angles = np.cos(actions)
    # none_actions = angles > 0.7
    # angles[none_actions] = 0

    every_hundredth = np.arange(98, 3000, 100)
    gradients = np.gradient(angles, axis=0)
    none_actions = abs(gradients) < 0.15
    gradients[none_actions] = 0
    added_gradients_across_digits = np.sum(gradients, axis=1)

    added_gradients = np.sum(gradients, axis=0)
    gradients_across_digits = np.gradient(angles, axis=1)
    all_gradients = gradients + gradients_across_digits
    difference_beginning_to_end = actions[0, :] - actions[-1, :]
    ratio_accumulative_change_to_total_change = added_gradients/difference_beginning_to_end

    distinct_grasps = activations_over_all_episodes[every_hundredth, :]

    pca = skld.PCA(3)
    pca.fit(distinct_grasps)
    X_pca = pca.transform(distinct_grasps)