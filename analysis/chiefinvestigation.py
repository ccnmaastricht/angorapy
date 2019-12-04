from agent.ppo import PPOAgent
from analysis.investigation import Investigator
import gym
import sklearn.manifold as sklm
import sklearn.decomposition as skld
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os

class Chiefinvestigator:

    def __init__(self, agent_id: int=1575396604, env_name: str="CartPole-v1"):
        """Chiefinvestigator can assign investigator to inspect the model and produce high-level analysis.

        Args:
            agent_id: ID of agent that will be analyzed.
            env_name: Name of the gym environment that the agent was trained in. Default is set to CartPole-v1
        """
        self.env = gym.make(env_name)
        self.new_agent = PPOAgent.from_agent_state(agent_id)  # agent id here # some agent: 1575396604 1575394142


    def parse_data(self):
        """Get state, activation, action, reward over one episode. Parse data to output

        Returns:
            activation_data, action_data, state_data, all_rewards
        """
        investi = Investigator(self.new_agent.policy)

        print(investi.list_layer_names())
        print(investi.list_layer_names()[2])
        activations_lstm = investi.get_activations_over_episode(investi.list_layer_names()[2], self.env, True)
    # print(activations_lstm)

# lengths, rewards = new_agent.evaluate(5, ray_already_initialized=True)

# activations
        activation_data = np.empty((len(np.array(activations_lstm)[:, 1]), 32))

        for k in range(len(np.array(activations_lstm)[:, 1])):
            activation_data[k, :] = np.array(activations_lstm)[k, 1]

        action_data = np.empty((len(np.array(activations_lstm)[:, 3]), 1))

        for m in range(len(np.array(activations_lstm)[:, 3])):
            action_data[m, :] = np.array(activations_lstm)[m, 3]
        action_data = action_data > 0

        #action_data = action_data < 0
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
        plt.scatter(results[:, 0], results[:, 1], c=action_data[:, 0])
        plt.title(title)
        plt.xlabel('second component')
        plt.ylabel('first component')
        plt.show()

if __name__ == "__main__":


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env_name = "CartPole-v1"
    agent_id: int = 1575469824 # 1575396604 # 1575469824  <- cartpole ffn 500 reward
    chiefinvesti = Chiefinvestigator(agent_id, env_name)
    x_activation_data, action_data, state_data, all_rewards = chiefinvesti.parse_data()


    zscores = sp.stats.zscore(x_activation_data) # normalization

    plt.plot(list(range(len(x_activation_data))), x_activation_data)
    plt.title('.')
    plt.show()

    # t-SNE
    RS = 12
    tsne_results = sklm.TSNE(random_state=RS).fit_transform(x_activation_data)
    # plot t-SNE results
    chiefinvesti.plot_results(tsne_results, action_data, "t-SNE")


    pca = skld.PCA(3)
    pca.fit(zscores)
    X_pca = pca.transform(zscores)
    chiefinvesti.plot_results(X_pca, action_data, "PCA")  # plot pca results

