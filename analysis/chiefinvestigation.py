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
        self.env = gym.make(env_name)
        self.new_agent = PPOAgent.from_agent_state(agent_id)  # agent id here # some agent: 1575396604 1575394142


    def parse_data(self):
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

        #action_data = action_data < 0
        # states
        state_data = np.empty((len(np.array(activations_lstm)[:, 0]), 4))
        for l in range(len(np.array(activations_lstm)[:, 0])):
            state_data[l, :] = np.array(activations_lstm)[l, 0]
        # rewards
        all_rewards = np.array(activations_lstm)[:, 2]

        return activation_data, action_data, state_data, all_rewards

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env_name = "CartPole-v1"
    agent_id: int = 1575396604
    chiefinvesti = Chiefinvestigator(agent_id, env_name)
    x_activation_data, action_data, state_data, all_rewards = chiefinvesti.parse_data()


    zscores = sp.stats.zscore(x_activation_data)

    plt.plot(list(range(len(x_activation_data))), x_activation_data)
    plt.show()
    RS = 123
    tsne_results = sklm.TSNE(random_state=RS).fit_transform(zscores)

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=action_data[:,0])
    plt.show()


    pca = skld.PCA(3)
    pca.fit(zscores)
    X_pca = pca.transform(zscores)
    # plot pca results
    plt.scatter(X_pca[:,0], X_pca[:,1], c=action_data[:,0])
    plt.show()


