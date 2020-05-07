import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from analysis.chiefinvestigation import Chiefinvestigator
from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stds, names, means, mean_durations, stds_duration = [], [], [], [],[]

for i, name in enumerate(["FF", "MF", "RF", "LF"]):
    env = f'HandFreeReach{name}Absolute-v0'

    agent_id = 1588151579 # small step reach task

    chiefinvesti = Chiefinvestigator(agent_id, env, from_iteration='best')

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)

    # collect data from episodes
    n_episodes = 20
    activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done = \
        chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

    done = np.vstack(done)
    first_success = []
    duration = []
    for d in done:
        try:
            first_success.append(np.argwhere(d)[0, 0])
            if len(np.argwhere(d)) == 200:
                duration.append(int(0))
            else:
                duration.append(len(np.argwhere(d)))
        except IndexError:
            first_success.append(int(200))
    print(first_success)
    std = np.std(np.vstack(first_success))
    std_duration = np.std(np.vstack(duration))
    print(std)
    stds.append(std)
    stds_duration.append(std_duration)
    means.append(np.mean(np.vstack(first_success)))
    mean_durations.append(np.mean(np.vstack(duration)))
    names.append(name)

x = np.arange(len(names))
width = 0.5

fig, ax = plt.subplots()
plt.bar(x, np.asarray(np.asarray(mean_durations)/mean_durations[3]), width)#, yerr=np.asarray(stds_duration), capsize= 4)

ax.set_ylabel('Duration of stay in target position')

ax.set_xticks(x)
ax.set_xticklabels(names)

plt.show()


