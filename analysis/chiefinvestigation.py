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

    def parse_data(self, layer_name: str):
        """Get state, activation, action, reward over one episode. Parse data to output.

        Args:
            layer_name: Name of layer to be analysed

        Returns:
            activation_data, action_data, state_data, all_rewards
        """
        investi = Investigator(self.new_agent.policy)
        activations_lstm = investi.get_activations_over_episode(layer_name, self.env, True)

        # lengths, rewards = new_agent.evaluate(5, ray_already_initialized=True)

        # activations
        activation_data = np.empty((len(np.array(activations_lstm)[:, 1]), 32))

        for k in range(len(np.array(activations_lstm)[:, 1])):
            activation_data[k, :] = np.array(activations_lstm)[k, 1]

        action_data = np.empty((len(np.array(activations_lstm)[:, 3]), 1))

        for m in range(len(np.array(activations_lstm)[:, 3])):
            action_data[m, :] = np.array(activations_lstm)[m, 3]
        action_data = action_data > 0

        # states
        state_data = np.empty((len(np.array(activations_lstm)[:, 0]), 4))
        for l in range(len(np.array(activations_lstm)[:, 0])):
            state_data[l, :] = np.array(activations_lstm)[l, 0]
        # rewards
        all_rewards = np.array(activations_lstm)[:, 2]

        return activation_data, action_data, state_data, all_rewards

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
    # 1575469824  <- cartpole ffn 500 reward
    # 1575394142  <- cartpole, recurrent 20 iterations trained
    # 1575396604 <- cartpole trained for two iterations, recurrent
    env_name = "CartPole-v1"
    agent_id: int = 1575394142
    chiefinvesti = Chiefinvestigator(agent_id, env_name)
    layer_names = chiefinvesti.print_layer_names()

    x_activation_data, action_data, state_data, all_rewards = chiefinvesti.parse_data(layer_names[2])

    zscores = sp.stats.zscore(x_activation_data) # normalization

    plt.plot(list(range(len(x_activation_data))), x_activation_data)
    plt.title('.')
    plt.show()

    # t-SNE
    RS = 12
    tsne_results = sklm.TSNE(random_state=RS).fit_transform(zscores)
    # plot t-SNE results
    plt.figure()
    chiefinvesti.plot_results(tsne_results, action_data[:, 0], "t-SNE")

    pca = skld.PCA(3)
    pca.fit(zscores)
    X_pca = pca.transform(zscores)
    plt.figure()
    chiefinvesti.plot_results(X_pca, action_data[:, 0], "PCA")  # plot pca results

    # chiefinvesti.plot_rewards(all_rewards)

    # loadings plot
    fig = plt.figure()
    plt.scatter(pca.components_[1, :], pca.components_[0, :])
    plt.title('Loadings plot')
    plt.show()


    x_train = x_activation_data[0:130, :]
    x_test = x_activation_data[-60:-1,:]
    y_train = action_data[0:130, :]
    y_test = action_data[-60:-1, :]

# classify actions based on activation in lstm using SVM
    from sklearn.svm import SVC

    clf = SVC(gamma='auto')
    clf.fit(x_train, y_train[:, 0])
    clf.predict(x_test)
    score = clf.score(x_test, y_test[:, 0])
    print(score)
    # timestepwise pca
    chiefinvesti.timestepwise_pca(x_activation_data, action_data, 'PCA')