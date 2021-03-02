import os
import sys

import autograd.numpy as np
from tqdm import tqdm
import gym
import sklearn

# sys.path.append("/dexterous-robot-hand/analysis/rnn_dynamical_systems")
from agent.ppo import PPOAgent
from analysis.investigation import Investigator
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper
from utilities.util import flatten
from utilities.model_utils import build_sub_model_from, build_sub_model_to


class Chiefinvestigator(Investigator):

    def __init__(self, agent_id: int, enforce_env_name: str = None, from_iteration='best'):
        """Chiefinvestigator can assign investigator to inspect the model and produce high-level analysis.

        Args:
            agent_id: ID of agent that will be analyzed.
            enforce_env_name: Name of the gym environment that the agent was trained in. Default is set to CartPole-v1
        """

        self.agent = PPOAgent.from_agent_state(agent_id, from_iteration=from_iteration)
        super().__init__(self.agent.policy, self.agent.distribution, self.agent.preprocessor)
        self.env = self.agent.env
        if enforce_env_name is not None:
            print(f"Enforcing environment {enforce_env_name} over agents original environment. If you want to use "
                  f"the same environment as the original agent anyways, there is no need to specify it in the"
                  f"constructor!")
            self.env = gym.make(enforce_env_name)
        self.agent.preprocessor = CombiWrapper([StateNormalizationWrapper(self.agent.state_dim),
                                                RewardNormalizationWrapper()])  # dirty fix, TODO remove soon
        self.weights = self.get_layer_weights('policy_recurrent_layer')
        self.n_hidden = self.weights[1].shape[0]
        self._get_rnn_type()
        try:
            self.sub_model_from = build_sub_model_from(self.network, "beta_action_head")
        except:
            try:
                self.sub_model_from = build_sub_model_from(self.network, 'discrete_action_head')
            except:
                self.sub_model_from = build_sub_model_from(self.network, 'gaussian_action_head')

        layer_names = self.get_layer_names()
        self.sub_model_to = build_sub_model_to(self.network, ['policy_recurrent_layer', layer_names[1]], include_original=True)

    def _get_rnn_type(self):
        '''Detect the rnn architecture in an agent'''
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
        states, activations, rewards, actions, done = self.get_activations_over_episode(
            [layer_name, previous_layer_name],
            self.env, False)

        # merge activations per layer into matrices where first dimension are timesteps
        activations = list(map(np.array, activations))

        return states, activations, rewards, actions, done

    def get_data_over_episodes(self, n_episodes: int, layer_name: str, previous_layer_name: str):
        """Get recorded data from given number of episodes

        Args:
            n_episodes: Number of episodes to be recorded
            layer_name: Name of recurrent layer
            previous_layer_name: Name of layer feeding into recurrent layer

        Returns: activations, inputs, actions, observations, info"""
        A, I, U, S, d = [], [], [], [], []
        for _ in tqdm(range(n_episodes)):
            s, a, r, u, done = self.parse_data(layer_name, previous_layer_name)

            A.append(np.reshape(a[1], (a[1].shape[0], self.n_hidden)))
            I.append(np.reshape(a[2], (a[2].shape[0], self.n_hidden)))
            U.append(np.vstack(u))
            S.append(np.vstack(s))
            d.append(done)

        return A, I, U, S, d

    def get_data_over_single_run(self, layer_name: str, previous_layer_name: str):

        states, activations, rewards, actions, done = self.parse_data(layer_name, previous_layer_name)
        inputs = np.reshape(activations[2], (activations[2].shape[0], self.n_hidden))
        activations = np.reshape(activations[1], (activations[1].shape[0], self.n_hidden))
        actions = np.vstack(actions)

        return activations, inputs, actions

    def get_sindy_datasets(self, settings):

        # collect data from episodes
        print("GENERATING DATA...")
        A, I, U, S, _ \
            = self.get_data_over_episodes(settings['n_episodes'], "policy_recurrent_layer", self.get_layer_names()[1])
        print(f"SIMULATED {settings['n_episodes']} episodes to generate {len(np.vstack(A))} data points.")

        # create training and testing datasets
        training_size = int(len(I) * 0.8)
        episode_size = int(len(np.vstack(S)) / settings['n_episodes'])  # average measure, may be meaningless

        dX, dI, dS = [], [], []
        for a, i, s in zip(A, I, S):  # TODO: improve gradient computation
            dX.append(np.gradient(a, axis=0))
            dI.append(np.gradient(i, axis=0))
            dS.append(np.gradient(s, axis=0))

        training_data = {'x': A[:training_size],
                         'dx': dX[:training_size],
                         'i': I[:training_size],
                         'di': dI[:training_size],
                         'u': U[:training_size],
                         's': S[:training_size],
                         'ds': dS[:training_size],
                         'e_size': episode_size}
        testing_data = {'x': A[training_size:],
                        'dx': dX[training_size:],
                        'i': I[training_size:],
                        'di': dI[training_size:],
                        'u': U[training_size:],
                        's': S[training_size:],
                        'ds': dS[training_size:],
                        'e_size': episode_size}

        return training_data, testing_data

    def render_fixed_points(self, activations):
        '''Function to repeatedly apply and render the state approached by an action.'''
        self.env.reset()
        for i in range(100):
            self.env.render()
            activation = activations[0, i, :].reshape(1, 1, self.n_hidden)
            probabilities = flatten(self.sub_model_from.predict(activation))

            action = self.distribution.act_deterministic(*probabilities)
            _, _, _, _ = self.env.step(action)

    def return_vanilla(self, inputs):

        def build_vanilla(weights, inputs):
            weights, inputweights, b = weights[1], weights[0], weights[2]
            vanilla = lambda x: - x + np.tanh(x @ weights + inputs @ inputweights + b)
            return vanilla

        vanilla = build_vanilla(self.weights, inputs)
        return vanilla

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
            g_fun = lambda x: np.tanh(r_fun(x) * (x @ U_h) + g_projection_b)

            gru = lambda x: - x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)
            return gru

        gru = build_gru(self.weights, inputs, self.n_hidden)
        return gru

    def compute_quiver_data(self, inputs, activations, stepsize, i):

        def real_meshgrid():
            x, y, z = np.meshgrid(np.linspace(-6, 6, stepsize),
                                  np.linspace(-6, 6, stepsize),
                                  np.linspace(-6, 6, stepsize))
            x, y, z = x.ravel(), y.ravel(), z.ravel()
            all = np.vstack((x, y, z)).transpose()
            return all

        pca = sklearn.decomposition.PCA(3)

        meshed_activations = real_meshgrid()
        reverse_transformed = pca.inverse_transform(meshed_activations)

        if self.rnn_type == 'vanilla':
            vanilla = self.return_vanilla(inputs[i, :])
            phase_space = vanilla(reverse_transformed)
        elif self.rnn_type == 'gru':
            gru = self.return_gru(np.zeros(self.n_hidden))
            phase_space = gru(reverse_transformed)

        transformed_phase_space = pca.transform(phase_space)
        x, y, z = meshed_activations[:, 0], meshed_activations[:, 1], meshed_activations[:, 2]
        u, v, w = transformed_phase_space[:, 0], transformed_phase_space[:, 1], transformed_phase_space[:, 2]

        return x, y, z, u, v, w


if __name__ == "__main__":
    os.chdir("../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1585500821  # cartpole-v1
    # agent_id, env = 1614534937, 'CartPoleContinuous-v0'  # continuous agent
    # agent_id = 1585500821  # cartpole-v1
    agent_id = 1614610158  # 'CartPoleContinuous-v0'  # continuous agent, no normalization
    # agent_id = 1586597938# finger tapping

    chiefinvesti = Chiefinvestigator(agent_id)

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)

    # collect data from episodes
    n_episodes = 1
    activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done \
        = chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

    import matplotlib.pyplot as plt
    a = ['x', 'x_dot', 'theta', 'theta_dot']
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(states_all_episodes[:, i])
        plt.title(a[i])
    plt.show()

