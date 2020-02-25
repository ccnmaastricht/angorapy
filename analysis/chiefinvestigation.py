import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.decomposition as skld
from matplotlib import animation
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from sklearn.svm import SVC
import sklearn.manifold as sklm
from mpl_toolkits.mplot3d import Axes3D

from agent.ppo import PPOAgent
from analysis.investigation import Investigator
# from utilities.util import insert_unknown_shape_dimensions
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper
from fixedpointfinder.fixedpointfinder import Adamfixedpointfinder
from mayavi import mlab
from fixedpointfinder.plot_utils import plot_fixed_points, plot_velocities


class Chiefinvestigator:

    def __init__(self, agent_id: int, enforce_env_name: str = None):
        """Chiefinvestigator can assign investigator to inspect the model and produce high-level analysis.

        Args:
            agent_id: ID of agent that will be analyzed.
            env_name: Name of the gym environment that the agent was trained in. Default is set to CartPole-v1
        """
        self.agent = PPOAgent.from_agent_state(agent_id)
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

    def plot_results(self, results, action_data, fixed_point, title: str = "Results", dreiD: bool = False):
        """Plot results of analysis performed by chiefinvestigator

        Args:
            results: Results of analysis to be plotted.
            action_data: Actions selected by agent that are used to color the plots.
            title: Title of the plot. Default is set to "Results"
            :param dreiD:
            :param fixed_point:

        Returns:
            Plot of data to be visualized

        """
        if dreiD is True:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                       marker='o', s=5, c=action_data[:])
            if fixed_point is not None:
                ax.scatter(fixed_point[:, 0], fixed_point[:, 1], fixed_point[:, 2],
                           marker='x', s=30)
            plt.title(title)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.show()

        if dreiD is False:
            plt.figure()
            plt.scatter(results[:, 1], results[:, 0],
                        c=action_data[:],
                        label=action_data[:])
            #legend_elements = [Line2D([0], [0], marker='o', label='Action: 0', color='w',
             #                         markerfacecolor='y', markersize=10),
             #                  Line2D([0], [0], marker='o', label='Action: 1', color='w',
              #                        markerfacecolor='tab:purple', markersize=10)]
            #plt.legend(handles=legend_elements)
            plt.title(title)
            plt.xlabel('second component')
            plt.ylabel('first component')
        # plt.pause(0.5)
            plt.show()


    def plot_rewards(self, rewards):
        plt.plot(list(range(len(rewards))), rewards)
        plt.xlabel('Number of steps in episode')
        plt.ylabel('Numerical reward')
        plt.title('Reward over episode')
        plt.show()

    def timestepwise_pca(self, activation_data, action_data, title: str = "Results"):
        zscores = sp.stats.zscore(activation_data)  # normalization

        pca = skld.PCA(3)
        pca.fit(zscores)
        X_pca = pca.transform(zscores)

        fig = plt.figure()
        plt.xlim([-3, 4])
        plt.ylim([-7.5, 10.5])

        def update(i):
            chiefinvesti.plot_results(X_pca[i:(i + 10), :], action_data[i:(i + 10), 0], title)  # plot pca results

        anim = animation.FuncAnimation(fig, update, frames=int(len(X_pca) - 1), interval=100)
        anim.save("moving_pca.gif",
                  writer='pillow')

    def svm_classifier(self, x_activation_data, action_data):
        size_train = 0.7
        x_train = x_activation_data[0:int(size_train*x_activation_data.shape[0]), :]
        x_test = x_activation_data[-int(size_train*x_activation_data.shape[0]):-1, :]
        y_train = action_data[0:int(size_train*x_activation_data.shape[0])]
        y_test = action_data[-int(size_train*x_activation_data.shape[0]):-1]

        clf = SVC(gamma='auto')
        clf.fit(x_train, y_train)
        clf.predict(x_test)
        score = clf.score(x_test, y_test)
        print("Classification accuracy", score)

    def nmfactor(self, activation):
        minimum = np.min(activation)
        activation = activation - minimum
        model = NMF(n_components=2)
        W = model.fit_transform(activation)
        H = model.components_
        return W, H


#  TODO: pca of whole activation over episode -> perhaps attempt to set in context of states
# -> construct small amounts of points (perhaps belonging to one action into
# -> dimensionality reduced space -> add them over time and see where they end up
# TODO: investigate role of momentum! take change along a principle component over time and see how that associates with action
# same for t-SNE
# TODO: apply unsupervised clustering stuff on the results of these techniques and the dataset -> through time stuff
# TODO: build neural network to predict grasp -> we trained a simple prediction model fp(z) containing one hidden
# layer with 64 units and ReLU activation, followed by a sigmoid output.
# perhaps even environmental factors
# TODO: look at weights
# TODO: dynamical systems and eigengrasps
# TODO: sequence analysis ideas -> sequence pattern and so forth:
# one possibility could be sequence alignment, time series analysis (datacamp)

if __name__ == "__main__":
    #os.chdir("../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    agent_id = 1578664065  # 1576849128 # 1576692646 from tonio
    chiefinvesti = Chiefinvestigator(agent_id)

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)

    activ = chiefinvesti.slave_investigator.get_layer_activations(layer_names[3])
    activationss, inputss, actionss = [], [], []
    for i in range(20):
        states, activations, rewards, actions = chiefinvesti.parse_data(layer_names[1],
                                                                        "policy_recurrent_layer")
        inputs = np.reshape(activations[2], (activations[2].shape[0], 64))
        activations = np.reshape(activations[1], (activations[1].shape[0], 64))
        activationss.append(activations)
        inputss.append(inputs)
        actions = np.vstack(actions)
        actionss.append(actions)
    weights = chiefinvesti.slave_investigator.get_layer_weights('policy_recurrent_layer')
    activationss, inputss = np.vstack(activationss), np.vstack(inputss)
    actionss = np.concatenate(actionss, axis=0)

    # adamfpf = Adamfixedpointfinder(weights, 'vanilla',
                               #    q_threshold=1e-02,
                               #    epsilon=0.01,
                               #    alr_decayr=1e-05,
                               #    max_iters=5000)
    #states = adamfpf.sample_states(activationss, 10)
    #input = np.zeros((states.shape[0], 64))
    #fps = adamfpf.find_fixed_points(states, input)
    #vel = adamfpf.compute_velocities(activationss, inputss)
    # plot_fixed_points(activationss, fps, 4000, 4)
    # plot_velocities(activationss, actionss[:, 0], 3000)
    idx = actionss[:, 1] < 0
    actionss[idx, 1] = 0
    plot_velocities(activationss, actionss[:, 1], 3000)





