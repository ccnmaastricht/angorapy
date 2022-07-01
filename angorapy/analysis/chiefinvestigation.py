import os
import gym
#import numpy as np
from time import sleep
import sys
sys.path.append("/Users/Raphael/angorapy/rnn_dynamical_systems")
import matplotlib.pyplot as plt
import sklearn

from rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points
from agent.ppo_agent import PPOAgent
from analysis.investigation import Investigator
from common.wrappers_old import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper
from utilities.util import flatten
from utilities.model_utils import build_sub_model_from
import autograd.numpy as np
import sklearn.decomposition as skld


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
        self.sub_model_from = build_sub_model_from(self.network, "beta_action_head")

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
        """Get state, activation, step_tuple, reward over one episode. Parse data to output.

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

    def return_gru(self, inputs):

        def build_gru(weights, input, n_hidden):
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden), np.arange(2 * n_hidden, 3 * n_hidden)
            W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
            U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
            b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

            z_projection_b = input @ W_z + b_z
            r_projection_b = input @ W_r + b_r
            g_projection_b = input @ W_h + b_h

            z_fun = lambda x: sigmoid(x @ U_z + z_projection_b)
            r_fun = lambda x: sigmoid(x @ U_r + r_projection_b)
            g_fun = lambda x: np.tanh((r_fun(x) * x) @ U_h + g_projection_b)

            gru = lambda x: - x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)
            return gru

        gru = build_gru(self.weights, inputs, self.n_hidden)
        return gru

    def return_vanilla(self, inputs):

        def build_vanilla(weights, inputs):
            weights, inputweights, b = weights[1], weights[0], weights[2]
            vanilla = lambda x: - x + np.tanh(x @ weights + inputs @ inputweights + b)
            return vanilla

        vanilla = build_vanilla(self.weights, inputs)
        return vanilla

    def compute_quiver_data(self, inputs, activations, timesteps, stepsize, i):

        def real_meshgrid(data_transformed):
            indices = np.arange(timesteps[0], timesteps[1], stepsize)
            x, y, z = np.meshgrid(data_transformed[indices, 0],
                                  data_transformed[indices, 1],
                                  data_transformed[indices, 2])
            x, y, z = np.meshgrid(np.linspace(-2, 2, stepsize),
                                  np.linspace(-2, 2, stepsize),
                                  np.linspace(-2, 2, stepsize))
            x, y, z = x.ravel(), y.ravel(), z.ravel()
            all = np.vstack((x, y, z)).transpose()
            return all

        pca = sklearn.decomposition.PCA(3)

        transformed_activations = pca.fit_transform(activations)
        meshed_activations = real_meshgrid(transformed_activations)
        reverse_transformed = pca.inverse_transform(meshed_activations)

        #transformed_input = pca.transform(inputs)
        #meshed_inputs = real_meshgrid(transformed_input)
        #reversed_input = pca.inverse_transform(meshed_inputs)

        if self.rnn_type == 'vanilla':
            vanilla = self.return_vanilla(inputs[i, :])
            phase_space = vanilla(reverse_transformed)
        elif self.rnn_type == 'gru':
            gru = self.return_gru(inputs[i, :])
            phase_space = gru(reverse_transformed)

        transformed_phase_space = pca.transform(phase_space)
        x, y, z = meshed_activations[:, 0], meshed_activations[:, 1], meshed_activations[:, 2]
        # x,y,z = transformed_activations[indices, 0], transformed_activations[indices, 1], transformed_activations[indices, 2]
        u, v, w = transformed_phase_space[:, 0], transformed_phase_space[:, 1], transformed_phase_space[:, 2]

        return x, y, z, u, v, w, transformed_activations[timesteps[0]:timesteps[1], :]


if __name__ == "__main__":
    os.chdir("../../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1583404415  # 1583180664 lunarlandercont
    # agent_id = 1585500821 # cartpole-v1
    # agent_id = 1585557832 # MountainCar # 1585561817 continuous
    agent_id = 1583256614 # reach task
    # agent_id = 1585777856# free reach
    chiefinvesti = Chiefinvestigator(agent_id)

    layer_names = chiefinvesti.get_layer_names()
    gru_weights = chiefinvesti.weights # weights are a list inputweights, recurrent weights and biases

    # collect data from episodes
    n_episodes = 5
    activations_over_all_episodes, inputs_over_all_episodes, \
    actions_over_all_episodes = chiefinvesti.get_data_over_episodes(n_episodes,
                                                                    "policy_recurrent_layer",
                                                                    layer_names[1])
    activations_single_run, inputs_single_run, actions_single_run = chiefinvesti.get_data_over_single_run('policy_recurrent_layer',
                                                                                                          layer_names[1])
    print('Standard deviation:', np.std(inputs_single_run))

    # employ fixedpointfinder
    adamfpf = Adamfixedpointfinder(chiefinvesti.weights, chiefinvesti.rnn_type,
                                   q_threshold=1e-12,
                                   tol_unique=1e-03,
                                   epsilon=0.1,
                                   alr_decayr=5e-03,
                                   agnc_normclip=2,
                                   agnc_decayr=1e-03,
                                   max_iters=5000)
    n_samples, noise_level = 100, 0
    states, sampled_inputs = adamfpf.sample_inputs_and_states(activations_over_all_episodes,
                                                              inputs_over_all_episodes,
                                                              n_samples,
                                                              noise_level)

    fps = adamfpf.find_fixed_points(states, sampled_inputs)

    # handle means of inputs
    mean_inputs = np.repeat(np.mean(inputs_over_all_episodes, axis=0).reshape(1, 32), repeats=n_samples, axis=0)
    fp_mean_input = adamfpf.find_fixed_points(states, mean_inputs)

    pca = skld.PCA(3)
    transformed_activations = pca.fit_transform(activations_over_all_episodes)
    fp_mean_input_transformed = pca.transform(fp_mean_input[0]['x'].reshape(1, -1))

    fp_locations = np.vstack([fp['x'] for fp in fps])
    mean_locations = np.mean(fp_locations, axis=0)
    mean_fp_transformed = pca.transform(mean_locations.reshape(1, -1))
    # plotting
    for i in range(100):
        fig, ax = plot_fixed_points(activations_single_run, fps, 4000, 1)

        timespan, stepsize = (0, 100), 3
        x, y, z, u, v, w, activations = chiefinvesti.compute_quiver_data(inputs_over_all_episodes, activations_over_all_episodes,
                                                                         timespan,
                                                                         stepsize, i)
        # mean fp will be red
        ax.scatter(fp_mean_input_transformed[:, 0], fp_mean_input_transformed[:, 1], fp_mean_input_transformed[:, 2], color='r')
        #ax.scatter(mean_fp_transformed[:, 0], mean_fp_transformed[:, 1], mean_fp_transformed[:, 2],
        #           color='m')
        ax.quiver(x, y, z, u, v, w, length=0.5, color='g')
        # ax.streamplot(x, y, z, u, v, w, color='g')


        plt.show()
        sleep(0.3)

    # loadings plot
    loadings = pca.components_
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(loadings[0, :], loadings[1, :], loadings[2, :])
    #plt.show()

    #import mayavi
    #from mayavi.mlab import quiver3d, plot3d
    #quiver3d(x, y, z, u, v, w)
    #plot3d(activations[:, 0], activations[:, 1], activations[:, 2])
    #mayavi.mlab.show()


    # render fixedpoints
    #for fp in fps:
    #    repeated_fps = np.repeat(fp['x'].reshape(1, 1, chiefinvesti.n_hidden), 100, axis=1)
    #    print("RENDERING FIXED POINT")
    #    chiefinvesti.render_fixed_points(repeated_fps)
    #    sleep(5)
