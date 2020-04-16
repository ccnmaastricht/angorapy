import os
import gym
import sys
sys.path.append("/Users/Raphael/dexterous-robot-hand/rnn_dynamical_systems")
import autograd.numpy as np
import matplotlib.pyplot as plt

from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder, Scipyfixedpointfinder, \
    AdamCircularFpf
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points
from analysis.rnn_dynamical_systems.fixedpointfinder.minimization import Minimizer
from analysis.chiefinvestigation import Chiefinvestigator

if __name__ == "__main__":
    os.chdir("../../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    agent_id, env = 1585777856, "HandFreeReachLFAbsolute-v0" # free reach


    chiefinvesti = Chiefinvestigator(agent_id, env)

    layer_names = chiefinvesti.get_layer_names()
    print(layer_names)
    # collect data from episodes
    n_episodes = 5
    activations_over_all_episodes, inputs_over_all_episodes, \
    actions_over_all_episodes, states_all_episodes = chiefinvesti.get_data_over_episodes(n_episodes,
                                                                                         "policy_recurrent_layer",
                                                                                         layer_names[1])
    activations_single_run, inputs_single_run, actions_single_run = chiefinvesti.get_data_over_single_run(
        'policy_recurrent_layer',
        layer_names[1])

    obs_model = states_all_episodes[1:100, :]
    actions_model = actions_over_all_episodes[:99, :]
    initial_weights = np.random.randn(68, 20) * 1e-03
    minmizer = Minimizer(epsilon=1e-02,
                         alr_decayr=1e-03,
                         init_agnc=1,
                         max_iter=30000)
    objective_fun = lambda x: np.mean(np.sum(np.square(actions_model - obs_model @ x), axis=1))
    trained_weights = minmizer.adam_optimization(objective_fun, initial_weights)

    circfpf = AdamCircularFpf(chiefinvesti.weights, chiefinvesti.rnn_type,
                              chiefinvesti.env, chiefinvesti.sub_model_to.get_weights(),
                              chiefinvesti.sub_model_from.get_weights(), trained_weights,
                              q_threshold=1e-02,
                              max_iters=10000,
                              epsilon=1e-02)

    states, inputs = circfpf.sample_inputs_and_states(activations_single_run, inputs_single_run,
                                                      20, 0)

    fps = circfpf.find_fixed_points(states)

    plot_fixed_points(activations_single_run, fps, 100, 1)
    plt.show()