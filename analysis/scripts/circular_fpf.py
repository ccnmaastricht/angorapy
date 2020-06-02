import os
import gym
import sys

sys.path.append("/Users/Raphael/dexterous-robot-hand/rnn_dynamical_systems")
import autograd.numpy as np
import matplotlib.pyplot as plt

from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder, \
    Scipyfixedpointfinder, \
    AdamCircularFpf
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points
from analysis.rnn_dynamical_systems.fixedpointfinder.minimization import Minimizer
from analysis.chiefinvestigation import Chiefinvestigator
import scipy as sc


def regress(Y, X, l=0.):
    '''
Parameters
----------
Y : floating point array (observations-by-outcomes)
    outcome variables
X : floating pint array (observation-by-predictors)
    predictors
l : float
    (optional) ridge penalty parameter
​
Returns
-------
beta : floating point array (predictors-by-outcomes)
    beta coefficients
'''

    if X.ndim > 1:
        n_observations, n_predictors = X.shape

    else:
        n_observations = X.size
        n_predictors = 1

    if n_observations < n_predictors:
        U, D, V = np.linalg.svd(X, full_matrices=False)

        D = np.diag(D)
        beta = np.matmul(
            np.matmul(
                np.matmul(
                    np.matmul(
                        V.transpose(),
                        sc.linalg.inv(
                            D ** 2 +
                            l * np.eye(n_observations))),
                    D),
                U.transpose()), Y)
    else:
        beta = np.matmul(
            np.matmul(
                sc.linalg.inv(
                    np.matmul(X.transpose(), X) +
                    l * np.eye(n_predictors)),
                X.transpose()), Y)

    return beta


if __name__ == "__main__":
    os.chdir("../../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    agent_id, env = 1585777856, "HandFreeReachLFAbsolute-v0"  # free reach

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
    for i in range(40):
        obs_model = states_all_episodes[1:100, :63]
        actions_model = actions_over_all_episodes[:99, :]
        trained_weights = regress(obs_model, actions_model, l=40)
        print(np.mean(np.sum(np.square(obs_model - actions_model @ trained_weights), axis=1)))

        test_set = states_all_episodes[101:200, :63]
        x_test = actions_over_all_episodes[101:200, :]
        print('test', np.mean(np.sum(np.square(test_set - x_test @ trained_weights), axis=1)))

    np.mean(np.sum(np.square(actions_model - obs_model @ trained_weights), axis=1))
    circfpf = AdamCircularFpf(chiefinvesti.weights, chiefinvesti.rnn_type,
                              chiefinvesti.env, chiefinvesti.sub_model_to.get_weights(),
                              chiefinvesti.sub_model_from.get_weights(), trained_weights,
                              q_threshold=1e-02,
                              max_iters=7000,
                              epsilon=1e-03)

    states, inputs = circfpf.sample_inputs_and_states(activations_single_run, inputs_single_run,
                                                      10, 0)

    fps = circfpf.find_fixed_points(states)

    plot_fixed_points(activations_single_run, fps, 100, 1)
    plt.show()
