import os
import gym
#import numpy as np
from time import sleep
import sys
sys.path.append("/Users/Raphael/dexterous-robot-hand/rnn_dynamical_systems")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder, AdamCircularFpf
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points
from agent.ppo import PPOAgent
from analysis.investigation import Investigator
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper
from utilities.util import parse_state, add_state_dims, flatten, insert_unknown_shape_dimensions
from utilities.model_utils import build_sub_model_from, build_sub_model_to
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
        try:
            self.sub_model_from = build_sub_model_from(self.network, "beta_action_head")
        except:
            self.sub_model_from = build_sub_model_from(self.network, 'discrete_action_head')

        layer_names = self.get_layer_names()
        self.sub_model_to = build_sub_model_to(self.network, ['policy_recurrent_layer', layer_names[1]], include_original=True)

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
        states_all_episodes = []
        for i in range(n_episodes):
            states, activations, rewards, actions = self.parse_data(layer_name, previous_layer_name)
            inputs = np.reshape(activations[2], (activations[2].shape[0], self.n_hidden))
            activations = np.reshape(activations[1], (activations[1].shape[0], self.n_hidden))
            activations_over_all_episodes.append(activations)
            inputs_over_all_episodes.append(inputs)
            actions = np.vstack(actions)
            actions_over_all_episodes.append(actions)
            states_all_episodes.append(np.vstack(states))

        activations_over_all_episodes, inputs_over_all_episodes = np.vstack(activations_over_all_episodes), \
                                                                  np.vstack(inputs_over_all_episodes)
        actions_over_all_episodes = np.concatenate(actions_over_all_episodes, axis=0)

        return activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes

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
    os.chdir("../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1583404415  # 1583180664 lunarlandercont
    # agent_id = 1585500821 # cartpole-v1
    # agent_id = 1585557832 # MountainCar # 1585561817 continuous
    # agent_id = 1583256614 # reach task
    agent_id = 1585777856 # free reach
    chiefinvesti = Chiefinvestigator(agent_id)

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)
    # collect data from episodes
    n_episodes = 5
    activations_over_all_episodes, inputs_over_all_episodes, \
    actions_over_all_episodes, states_all_episodes = chiefinvesti.get_data_over_episodes(n_episodes,
                                                                    "policy_recurrent_layer",
                                                                    layer_names[1])
    activations_single_run, inputs_single_run, actions_single_run = chiefinvesti.get_data_over_single_run('policy_recurrent_layer',
                                                                                                          layer_names[1])


    def build_numpy_submodelto(submodel_weights):
        first_layer = lambda x: np.tanh(x @ submodel_weights[0] + submodel_weights[1])
        second_layer = lambda x: np.tanh(x @ submodel_weights[2] + submodel_weights[3])
        third_layer = lambda x: np.tanh(x @ submodel_weights[4] + submodel_weights[5])
        fourth_layer = lambda x: np.tanh(x @ submodel_weights[6] + submodel_weights[7])

        return first_layer, second_layer, third_layer, fourth_layer

    def build_numpy_submodelfrom(submodel_weights):
        softplus = lambda x: np.log(np.exp(x) + 1)
        alpha_fun = lambda x: softplus(x @ submodel_weights[0] + submodel_weights[1])
        beta_fun = lambda x: softplus(x @ submodel_weights[2] + submodel_weights[3])

        return alpha_fun, beta_fun

    submodelfrom_weights = chiefinvesti.sub_model_from.get_weights()
    alpha_fun, beta_fun = build_numpy_submodelfrom(submodelfrom_weights)
    alphas = alpha_fun(activations_over_all_episodes)
    betas = beta_fun(activations_over_all_episodes)


    def act_deterministic(alphas, betas):
        actions = (alphas - 1) / (alphas + betas - 2)

        action_max_values = chiefinvesti.env.action_space.high
        action_min_values = chiefinvesti.env.action_space.low
        action_mm_diff = action_max_values - action_min_values

        actions = np.multiply(actions, action_mm_diff) + action_min_values

        return actions
    actions = act_deterministic(alphas, betas)

    def unwrapper(means, variances, states):
        unwrapped_states = []
        for state in states:
            unwrapped_states.append((state * np.sqrt(variances)) + means)

        return np.vstack(unwrapped_states)

    means = chiefinvesti.preprocessor.wrappers[0].mean[0]
    variances = chiefinvesti.preprocessor.wrappers[0].variance[0]
    states_all_episodes = np.vstack(states_all_episodes)
    unwrapped_states = unwrapper(means, variances, states_all_episodes)


    pca = skld.PCA(3)

    transformed_states = pca.fit_transform(unwrapped_states[:100, :])
    transformed_states_wrapped = pca.transform(states_all_episodes[:100, :])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(transformed_states[:, 0], transformed_states[:, 1], transformed_states[:, 2])
    plt.title('unwrapped states')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(transformed_states_wrapped[:, 0], transformed_states_wrapped[:, 1], transformed_states_wrapped[:, 2])
    plt.title('wrapped states')
    plt.show()

    # actual_action = actuation_center + aoi * actuation_range
    geom_quat = chiefinvesti.env.sim.model.geom_quat
    # finding orientation and angle of geoms
    some_quat = geom_quat[9, :]
    a_some_quat = some_quat[1:] / np.sqrt(np.sum(np.square(some_quat[1:])))
    print(a_some_quat)
    theta_some_quat = 2 * np.arctan2(np.sqrt(np.sum(np.square(some_quat[1]))), some_quat[0])
    print(np.degrees(theta_some_quat))

    # manipulation of geoms according to actions
    new_angle = actual_action[0] - theta_some_quat

    # update quaternion
    # new_quat = np.ndarray((np.cos(new_angle/2), 0, np.sin(new_angle/2)*some_quat[2], 0))

    def quaternion_mult(q, r):
        return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
                r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
                r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
                r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


    def point_rotation_by_quaternion(point, q):
        r = point
        q_conj = [q[0], -1 * q[1], -1 * q[2], -1 * q[3]]
        return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]


    rotation_quat = [np.cos(actual_action[3] / 2), np.sin(actual_action[3] / 2), 0, 0]
    print(point_rotation_by_quaternion(some_quat, rotation_quat))
    a = point_rotation_by_quaternion(some_quat, rotation_quat)

    print(np.degrees(angle_between(a, a_some_quat)))
    print(np.degrees(actual_action[3]))




    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)


    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))