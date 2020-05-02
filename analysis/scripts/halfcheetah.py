import os
import sys
import matplotlib.pyplot as plt

from analysis.chiefinvestigation import Chiefinvestigator
from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

agent_id = 1588341681 # halfcheetah

chiefinvesti = Chiefinvestigator(agent_id)

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 5
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes = \
    chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

# employ fixedpointfinder
adamfpf = Adamfixedpointfinder(chiefinvesti.weights, chiefinvesti.rnn_type,
                               q_threshold=1e-05,
                               epsilon=5e-03)

states, inputs = adamfpf.sample_inputs_and_states(activations_over_all_episodes, inputs_over_all_episodes,
                                                  50, 0)
fps = adamfpf.find_fixed_points(states, inputs)

plot_fixed_points(activations_over_all_episodes, fps, 200, 1)
plt.show()