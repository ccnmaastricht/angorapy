import os

import gym
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import scipy as sp
import sklearn.decomposition as skld
from matplotlib import animation
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from scipy.optimize import minimize, curve_fit
from sklearn.decomposition import NMF
from sklearn.svm import SVC
import sklearn.manifold as sklm
from mpl_toolkits.mplot3d import Axes3D

from agent.ppo import PPOAgent
from analysis.investigation import Investigator
# from utilities.util import insert_unknown_shape_dimensions
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper


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

    def plot_results(self, results, action_data, title: str = "Results", dreiD: bool = False):
        """Plot results of analysis performed by chiefinvestigator

        Args:
            results: Results of analysis to be plotted.
            action_data: Actions selected by agent that are used to color the plots.
            title: Title of the plot. Default is set to "Results"

        Returns:
            Plot of data to be visualized
        """
        if dreiD is True:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                       marker='o', s=5, c=action_data[:])
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
        print(score)

    def minimiz(self, weights, inputweights, input, activation, method: str = 'trust-krylov'):
        id = np.random.randint(activation.shape[0])
        print(id)
        x0 = activation[id, :]
        input = input[id, :]
        fun = lambda x: 0.5 * sum(
            (- x[0:64] + np.matmul(weights, np.tanh(x[0:64])) + np.matmul(inputweights, input)) ** 2)
        # der = lambda x: np.sum((- np.eye(64, 64) + weights * (1 - np.tanh(x[0:64]) ** 2)),
                            #   axis=1)  # - np.eye(64, 64) + weights*x[0:64]
        options = {'gtol': 1e-5, 'disp': True}
        # Jac = nd.Jacobian(fun)
        # print(Jac.shape)
        Hes = nd.Hessian(fun)
        print(Hes)
        y = fun(x0)
        print(y)
        optimisedResults = minimize(fun, x0, method=method, jac=False, hess=Hes,
                                    options=options)
        return optimisedResults

    def curve_fit_minimization(self, activation, weights, inputweights, input):
        id = np.random.randint(activation.shape[0])
        print(id)
        x0 = activation[id, :]
        x = activation[id, :]
        input = input[id, :]
        fun = lambda weights: 0.5 * sum(
            (- x[0:64] + np.matmul(weights, np.tanh(x[0:64])) + np.matmul(inputweights, input)) ** 2)
        ydata = np.zeros(activation.shape[1])
        popt, pcov = curve_fit(fun, xdata=weights, ydata=ydata, p0=x0)
        return popt, pcov

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

    agent_id: int = 1578664065  # 1576849128 # 1576692646 from tonio
    chiefinvesti = Chiefinvestigator(agent_id)

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)

    activ = chiefinvesti.slave_investigator.get_layer_activations(layer_names[3])
    states, activations, rewards, actions = chiefinvesti.parse_data(layer_names[1],
                                                                    "policy_recurrent_layer")
    actions = np.vstack(actions)
    # weights = chiefinvesti.slave_investigator.get_layer_weights('policy_recurrent_layer')
    # print(weights)
    # method = "trust-ncg"
    # optimiserResults = chiefinvesti.minimiz(weights=weights[1], inputweights=weights[0],
    #                                        input=previous_activations[0], activation=activations[1],
    #                                        method=method)

    zscores = sp.stats.zscore(np.reshape(activations[1], (activations[1].shape[0], 64))) # normalization

    # plt.plot(list(range(len(activations[1]))), np.reshape(activations[1], (activations[1].shape[0], 64)))
    # plt.title('Activations over episode')
    # plt.show()

    plt.imshow(np.transpose(np.reshape(activations[1], (activations[1].shape[0], 64))))
    plt.show()

    # NMF
    W, H = chiefinvesti.nmfactor(np.reshape(activations[1], (activations[1].shape[0], 64)))

    # t-SNE
    RS = 12
    tsne_results = sklm.TSNE(random_state=RS).fit_transform(zscores)
    # plot t-SNE results
    chiefinvesti.plot_results(tsne_results, actions[:, 1], "t-SNE", False)

    pca = skld.PCA(3)
    pca.fit(zscores)
    X_pca = pca.transform(zscores)
    chiefinvesti.plot_results(X_pca, actions[:, 1], "PCA", True)  # plot pca results
    chiefinvesti.plot_results(X_pca, actions[:, 1], "PCA", False)
    #new_pca = pca.transform(optimiserResults.x.reshape(1, -1))
    #chiefinvesti.plot_rewards(all_rewards)

    # loadings plot
    loadings_plot = False
    if loadings_plot is True:
        fig = plt.figure()
        plt.scatter(pca.components_[1, :], pca.components_[0, :])
        plt.title('Loadings plot')
        plt.show()

    svm_classify = True
    if svm_classify is True:
        action_data = actions[:, 1] > 0.5
        action_data_1 = actions[:, 1] < -0.5
        action_data = action_data + action_data_1
        chiefinvesti.svm_classifier(np.reshape(activations[1], (activations[1].shape[0], 64)), action_data)

    # timestepwise pca
    # chiefinvesti.timestepwise_pca(x_activation_data, action_data, 'PCA')

    # pca of actions to identify unique actions
    if loadings_plot is True:
        pca = skld.PCA(2)
        pca.fit(actions)
        action_pca = pca.transform(actions)
        plt.scatter(action_pca[:, 1], action_pca[:, 0])
        plt.title('PCA of action space')
        plt.show()
