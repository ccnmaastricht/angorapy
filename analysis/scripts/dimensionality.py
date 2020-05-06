import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition as skld
from analysis.rnn_dynamical_systems.fixedpointfinder.three_bit_flip_flop import Flipflopper

from analysis.chiefinvestigation import Chiefinvestigator


environments_agents = {'3-Bit-FlipFlop': 'gru',
                       'LunarLander': 1583404415,
                       'CartPole': 1585500821,
                       'MountainCar': 1585557832,
                       'HalfCheetah': 1588341681,
                       'ReachTask':  1588151579,
                       }

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_cumsum_explained_variance(activation):

    pca = skld.PCA(10)
    pca.fit_transform(activation)

    return np.cumsum(pca.explained_variance_ratio_)


fig = plt.figure()
ax = plt.axes()
for environment, agent in environments_agents.items():

    if environment == '3-Bit-FlipFlop':
        flopper = Flipflopper(rnn_type=agent, n_hidden=24)
        stim = flopper.generate_flipflop_trials()
        activations = flopper.get_activations(stim)

        cumsum_explained_variance = get_cumsum_explained_variance(np.vstack(activations))
        ax.plot(cumsum_explained_variance*100, label=environment)

    else:
        chiefinvesti = Chiefinvestigator(agent, from_iteration='best')
        layer_names = chiefinvesti.get_layer_names()
        n_episodes = 10
        activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done = \
            chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

        cumsum_explained_variance = get_cumsum_explained_variance(activations_over_all_episodes)
        ax.plot(cumsum_explained_variance*100, label=environment)

ax.set_xticks(range(0, 10))
ax.set_xticklabels(range(1, 11))
ax.legend()
ax.set_xlabel('Number of Principle Components')
ax.set_ylabel('% explained variance')
plt.show()