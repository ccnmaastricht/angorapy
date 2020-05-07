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

chiefinvesti = Chiefinvestigator(agent_id, env, from_iteration='best')

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 5
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done = \
    chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

done = np.vstack(done)
achieved = []
for d in done:
    achieved.append(np.argwhere(d)[0, 0])

episode_times = np.arange(n_episodes)*200+199
achieved_times = np.asarray(achieved)+np.arange(n_episodes)*200
#plt.plot(activations_over_all_episodes[:1000, :], linewidth=0.3)
#plt.xlabel('Timesteps')
#plt.ylabel('Activation')
#plt.show()
from time import sleep
#fig = plt.figure()

all_weights = chiefinvesti.network.get_weights()
variance_recurrent_units = np.var(activations_over_all_episodes[achieved_times:episode_times, :], axis=0)
alpha_weights = all_weights[11]
mean_per_recurrent_unit = np.abs(np.mean(alpha_weights, axis=1))

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(variance_recurrent_units.reshape(1, -1).T, np.abs(np.mean(alpha_weights, axis=1)))
plt.scatter(variance_recurrent_units, np.abs(np.mean(alpha_weights, axis=1)))
plt.plot(variance_recurrent_units, reg.predict(variance_recurrent_units.reshape(1, -1).T), c='k')
plt.xlabel('variance')
plt.ylabel('absolute mean weight')
plt.show()

for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.title(chiefinvesti.env.sim.model.actuator_id2name(i))
    plt.scatter(variance_recurrent_units, np.abs(alpha_weights[:, i]), s=0.6)
    #variance_over_weights = variance_recurrent_units * alpha_weights
    if i == 16:
        plt.xlabel('variance')
        plt.ylabel('weight')
plt.show()

import sklearn
pca = sklearn.decomposition.PCA(10)
pca.fit_transform(activations_over_all_episodes)
print(np.cumsum(pca.explained_variance_ratio_))