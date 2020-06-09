import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from analysis.chiefinvestigation import Chiefinvestigator

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def achieve_invest(agent_id, env):
    stds, names, means, mean_durations, stds_duration = [], [], [], [], []
    #all_mean_durations = np.zeros((4, 30))
    chiefinvesti = Chiefinvestigator(agent_id, env, from_iteration='best')

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)

    # collect data from episodes
    n_episodes = 30
    activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done = \
        chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

    done = np.vstack(done)
    first_success = []
    duration = []
    for k, d in enumerate(done):
        try:
            first_success.append(np.argwhere(d)[0, 0])
            if len(np.argwhere(d)) == 200:
                duration.append(int(0))
                #all_mean_durations[i, k] = 0
            else:
                duration.append(len(np.argwhere(d)))
               # all_mean_durations[i, k] = len(np.argwhere(d))
        except IndexError:
            first_success.append(int(200))

    std = np.std(np.vstack(first_success))
    #std_duration = np.std(np.vstack(duration))
    stds.append(std)
    #stds_duration.append(std_duration)
    means.append(np.mean(np.vstack(first_success)))
    mean_durations.append(np.mean(np.vstack(duration)))
    #names.append(name)

    return np.std(np.vstack(first_success)), np.mean(np.vstack(first_success))

names = ["FF", "MF", "RF", "LF"]
stds, means, means_single_goal, stds_single_goal = [], [], [], []

for i in range(8):
    if i < 4:
        env = f'HandFreeReach{names[i]}Absolute-v0'

        agent_id = 1588151579 # small step reach task
    elif i == 4:
        agent_id, env = 1588944848, 'HandFreeReachFFAbsolute-v0'  # single goal ff reach task
    elif i == 5:
        agent_id, env = 1591436665, 'HandFreeReachMFAbsolute-v0'  # single goal mf
    elif i == 6:
        agent_id, env = 1591525051, 'HandFreeReachRFAbsolute-v0' # single goal rf
    elif i == 7:
        agent_id, env = 1591604443, 'HandFreeReachLFAbsolute-v0'# single goal lf

    if agent_id == 1588151579:
        std, mean = achieve_invest(agent_id, env)
        stds.append(std)
        means.append(mean)
    else:
        std_single_goal, mean_single_goal = achieve_invest(agent_id, env)
        stds_single_goal.append(std_single_goal)
        means_single_goal.append(mean_single_goal)

# plot means and standard deviations of first successfully reaching the target
x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots()
plt.bar(x - width/2, np.asarray(means), width, yerr=np.asarray(stds), capsize=4, label='multi goal')
plt.bar(x+width/2, np.asarray(means_single_goal), width, yerr=np.asarray(stds_single_goal), capsize=4, label='single goal')

ax.set_ylabel('Timestep of first success')
ax.set_xticks(x)
ax.set_xticklabels(names)
plt.legend()
plt.show()

# plot durations that the agent stayed in target position
#plt.boxplot((all_mean_durations/np.repeat(np.asarray(mean_durations).reshape(-1,1), n_episodes, axis=1)).T*100, labels=names)
#plt.ylabel('% normalized duration in goal position')
#plt.show()


