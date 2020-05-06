import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from analysis.chiefinvestigation import Chiefinvestigator
from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

agent_id, env = 1588151579, 'HandFreeReachFFAbsolute-v0' # small step reach task

chiefinvesti = Chiefinvestigator(agent_id, from_iteration='best')

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 20
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done = \
    chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

all_weights = chiefinvesti.network.get_weights()
variance_recurrent_units = np.var(activations_over_all_episodes[104:, :], axis=0)
alpha_weights = all_weights[11]
mean_per_recurrent_unit = np.abs(np.mean(alpha_weights, axis=1))

from time import sleep
#fig = plt.figure()


for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.title(chiefinvesti.env.sim.model.actuator_id2name(i))
    plt.scatter(variance_recurrent_units, np.abs(alpha_weights[:, i]), s=0.6)
    #variance_over_weights = variance_recurrent_units * alpha_weights
    if i == 16:
        plt.xlabel('variance')
        plt.ylabel('weight')
plt.show()

plt.scatter(variance_recurrent_units, np.abs(np.mean(alpha_weights, axis=1)))
#plt.plot(reg)
plt.xlabel('variance')
plt.ylabel('absolute mean weight')
plt.show()

import sklearn
pca = sklearn.decomposition.PCA(10)
pca.fit_transform(activations_over_all_episodes)
print(np.cumsum(pca.explained_variance_ratio_))


# employ fixedpointfinder
#adamfpf = Adamfixedpointfinder(chiefinvesti.weights, chiefinvesti.rnn_type,
               #                q_threshold=3e-15,
               #                tol_unique=2e-03,
               #                epsilon=5e-03)

#states, inputs = adamfpf.sample_inputs_and_states(activations_over_all_episodes, inputs_over_all_episodes,
                      #                            100, 0)
#fps = adamfpf.find_fixed_points(states, inputs)

#plot_fixed_points(activations_over_all_episodes, fps, 200, 1)
#plt.show()

#for fp in fps:
#    chiefinvesti.render_fixed_points(np.repeat(np.reshape(fp['x'], (1, 1, 32)), axis=1, repeats=100))
#    sleep(3)


