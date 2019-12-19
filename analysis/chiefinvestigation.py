from agent.ppo import PPOAgent
from analysis.investigation import Investigator
import gym
import sklearn.manifold as sklm
import sklearn.decomposition as skld
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
from matplotlib.lines import Line2D
from matplotlib import animation
from sklearn.svm import SVC
from scipy.optimize import minimize
from tensorflow import matmul
import numdifftools as nd
#from utilities.util import insert_unknown_shape_dimensions
import tensorflow as tf

class Chiefinvestigator:

    def __init__(self, agent_id: int=1575396604, env_name: str="CartPole-v1"):
        """Chiefinvestigator can assign investigator to inspect the model and produce high-level analysis.

        Args:
            agent_id: ID of agent that will be analyzed.
            env_name: Name of the gym environment that the agent was trained in. Default is set to CartPole-v1
        """
        self.env = gym.make(env_name)
        self.new_agent = PPOAgent.from_agent_state(agent_id)

    def print_layer_names(self):
        investi = Investigator(self.new_agent.policy)

        print(investi.list_layer_names())
        return investi.list_layer_names()

    def parse_data(self, layer_name: str, previous_layer_name: str):
        """Get state, activation, action, reward over one episode. Parse data to output.

        Args:
            layer_name: Name of layer to be analysed

        Returns:
            activation_data, action_data, state_data, all_rewards
        """
        investi = Investigator(self.new_agent.policy)
        activations_lstm = investi.get_activations_over_episode(layer_name, previous_layer_name, self.env, True)
        states, activations, previous_activation, rewards, actions = map(lambda x: np.array(x), zip(*activations_lstm))
        activations = np.reshape(activations, (activations.shape[0], 64))
        previous_activation = np.reshape(previous_activation, (previous_activation.shape[0], 64))

        return states, activations, previous_activation, rewards, actions

    def return_weights(self, layer_name):
        investi = Investigator(self.new_agent.policy)
        return investi.get_layer_weights(layer_name)

    def plot_results(self, results, action_data, title: str="Results"):
        """Plot results of analysis performed by chiefinvestigator

        Args:
            results: Results of analysis to be plotted.
            action_data: Actions selected by agent that are used to color the plots.
            title: Title of the plot. Default is set to "Results"

        Returns:
            Plot of data to be visualized
        """
        plt.scatter(results[:, 1], results[:, 0],
                    c=action_data[:],
                    label=action_data[:])
        legend_elements = [Line2D([0], [0], marker='o', label='Action: 0', color='w',
                                  markerfacecolor='y', markersize=10),
                           Line2D([0], [0], marker='o', label='Action: 1', color='w',
                                  markerfacecolor='tab:purple', markersize=10)]
        plt.legend(handles=legend_elements)
        plt.title(title)
        plt.xlabel('second component')
        plt.ylabel('first component')
        #plt.pause(0.5)
        plt.show()


    def plot_rewards(self, rewards):
        plt.plot(list(range(len(rewards))), rewards)
        plt.xlabel('Number of steps in episode')
        plt.ylabel('Numerical reward')
        plt.title('Reward over episode')
        plt.show()

    def timestepwise_pca(self, activation_data, action_data, title: str="Results"):

        zscores = sp.stats.zscore(activation_data)  # normalization

        pca = skld.PCA(3)
        pca.fit(zscores)
        X_pca = pca.transform(zscores)

        fig = plt.figure()
        plt.xlim([-3,4])
        plt.ylim([-7.5, 10.5])
        def update(i):

            chiefinvesti.plot_results(X_pca[i:(i+10),:], action_data[i:(i+10), 0], title)  # plot pca results
        anim = animation.FuncAnimation(fig, update, frames = int(len(X_pca)-1), interval=100)
        anim.save("moving_pca.gif",
                writer='pillow')

    def svm_classifier(self, x_activation_data, action_data):
        x_train = x_activation_data[0:130, :]
        x_test = x_activation_data[-60:-1, :]
        y_train = action_data[0:130, :]
        y_test = action_data[-60:-1, :]
        # classify actions based on activation in lstm using SVM

        clf = SVC(gamma='auto')
        clf.fit(x_train, y_train[:, 0])
        clf.predict(x_test)
        score = clf.score(x_test, y_test[:, 0])
        print(score)

    def minimiz(self, weights, inputweights, input, activation, method: str='trust-krylov'):
        id = np.random.randint(activation.shape[0])
        print(id)
        x0 = activation[id, :]
        input = input[id, :]
        fun = lambda x: 0.5*sum((- x[0:64] + np.matmul(weights, np.tanh(x[0:64])) + np.matmul(inputweights, input))**2)
        # der = lambda x: np.sum((- np.eye(64, 64) + weights * (1-np.tanh(x[0:64])**2)), axis=1)# - np.eye(64, 64) + weights*x[0:64]
        options = {'gtol': 1e-5, 'disp': True}
        #Jac = nd.Jacobian(fun)
        #print(Jac.shape)
        Hes = nd.Hessian(fun)
        print(Hes)
        y = fun(x0)
        print(y)
        optimisedResults = minimize(fun, x0, method=method, jac=False, hess=Hes,
                                    options=options)

        return optimisedResults
#  TODO: pca of whole activation over episode -> perhaps attempt to set in context of states
                                     # -> construct small amounts of points (perhaps belonging to one action into
                                     # -> dimensionality reduced space -> add them over time and see where they end up
# TODO: investigate role of momentum! take change along a principle component over time and see how that associates with action
# same for t-SNE
# TODO: apply unsupervised clustering stuff on the results of these techniques and the dataset
# TODO: build neural network to predict grasp -> we trained a simple prediction model fp(z) containing one hidden
                                            # layer with 64 units and ReLU activation, followed by a sigmoid output.
# TODO: Do fixed points evolve where two clusters (actions) are? -> dynamical systems analysis
# TODO: is reward coupled to momentum ?
# TODO: look at weights
# TODO: dynamical systems and eigengrasps

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env_name = "LunarLanderContinuous-v2"
    agent_id: int = 1576655668 #1576489378 # 1575394142
    chiefinvesti = Chiefinvestigator(agent_id, env_name)
    layer_names = chiefinvesti.print_layer_names()
    new_agent = PPOAgent.from_agent_state(agent_id)

    inv = Investigator(new_agent.policy)

    # activ = inv.get_layer_activations(layer_names[3])
    states, activations, previous_activations, rewards, actions = chiefinvesti.parse_data("policy_recurrent_layer",
                                                                                          layer_names[2])
    weights = chiefinvesti.return_weights(layer_names[3])
    #print(weights)
    method = "trust-ncg"
    optimiserResults = chiefinvesti.minimiz(weights=weights[1], inputweights=weights[0],
                                            input=previous_activations, activation=activations,
                                            method=method)

    # zscores = sp.stats.zscore(activations) # normalization

    plt.plot(list(range(len(activations))), activations)
    plt.title('.')
    plt.show()

    # t-SNE
    #RS = 12
    #tsne_results = sklm.TSNE(random_state=RS).fit_transform(zscores)
    # plot t-SNE results
    #plt.figure()
    #chiefinvesti.plot_results(tsne_results, action_data[:, 0], "t-SNE")

    pca = skld.PCA(3)
    pca.fit(activations)
    X_pca = pca.transform(activations)
    plt.figure()
    chiefinvesti.plot_results(X_pca, actions[:, 0], "PCA")  # plot pca results
    new_pca = pca.transform(optimiserResults.x.reshape(1, -1))
    # chiefinvesti.plot_rewards(all_rewards)

    # loadings plot
    #fig = plt.figure()
    #plt.scatter(pca.components_[1, :], pca.components_[0, :])
    #plt.title('Loadings plot')
    #plt.show()

    # chiefinvesti.svm_classifier(x_activation_data, action_data)

    # timestepwise pca
    #chiefinvesti.timestepwise_pca(x_activation_data, action_data, 'PCA')

    # pca of actions to identify unique actions
    pca = skld.PCA(2)
    pca.fit(actions)
    action_pca = pca.transform(actions)
    plt.scatter(list(range(len(action_pca))), action_pca[:, 0])
    plt.show()