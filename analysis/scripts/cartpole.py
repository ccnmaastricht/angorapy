import os
import sys
import matplotlib.pyplot as plt

from analysis.chiefinvestigation import Chiefinvestigator
from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir("../../")  # remove if you want to search for ids in the analysis directory


# agent_id = 1585500821 # cartpole-v1
agent_id = 1590500032 # cartpole-v1

chiefinvesti = Chiefinvestigator(agent_id)

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 5
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done = \
    chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

# employ fixedpointfinder
adamfpf = Adamfixedpointfinder(chiefinvesti.weights, chiefinvesti.rnn_type,
                               epsilon=1e-02,
                               q_threshold=1e-12)

states, inputs = adamfpf.sample_inputs_and_states(activations_over_all_episodes, inputs_over_all_episodes,
                                                  400, 0)
fps = adamfpf.find_fixed_points(states, inputs)

fig, ax = plot_fixed_points(activations_over_all_episodes, fps, 1000, 1)
# add vector field to plot
x, y, z, u, v, w = chiefinvesti.compute_quiver_data(inputs_over_all_episodes, activations_over_all_episodes,
                                                    5, 5)
ax.quiver(x,y,z,u,v,w, color='g', length=0.3)
plt.show()