import os
import numpy as np
import jax.numpy as jnp
import sys
sys.path.append("/home/raphael/Code/dexterous-robot-hand/")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder import control_autoencoder, utils
from jax.experimental import optimizers
from jax import random
import sklearn.decomposition as skld
import time
import pickle
import argparse


def main(agent_id, n_networks):
    chiefinv = Chiefinvestigator(agent_id)

    SAVE_DIR = "analysis/sindy_autoencoder/storage/"
    training_data = pickle.load(open(SAVE_DIR + "training_data.pkl", "rb"))
    testing_data = pickle.load(open(SAVE_DIR + "testing_data.pkl", "rb"))
    n_episodes = 1
    activations_all_episodes, inputs_all_episodes, actions_all_episodes, states_all_episodes, _ \
        = chiefinv.get_data_over_episodes(n_episodes, "policy_recurrent_layer", chiefinv.get_layer_names()[1])

    # load trained system
    state = control_autoencoder.load_state("SindyControlAutoencoder")
    params, coefficient_mask = state['autoencoder'], state['coefficient_mask']
    control_autoencoder.plot_params(params, coefficient_mask)
    plt.show()

    # Simulate
    n_points = 1000 # 3 episodes
    [z, y, sindy_predict] = control_autoencoder.batch_compute_latent_space(params, coefficient_mask,
                                                                           activations_all_episodes, inputs_all_episodes)
    # Simulate Dynamics and produce 3 episodes
    _, _, simulated_activations, simulation_results, actions = control_autoencoder.simulate_episode(chiefinv,
                                                                                                    params, coefficient_mask,
                                                                                                    render=True)
    # Reduce Dimensions
    activation_pca = skld.PCA(3)
    X_activations = activation_pca.fit_transform(activations_all_episodes)
    reconstruction_pca = skld.PCA(3)
    X_reconstruction = reconstruction_pca.fit_transform(z)
    X_rec_simulation = reconstruction_pca.transform(simulation_results)
    X_act_simulation = activation_pca.transform(simulated_activations)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221, projection=Axes3D.name)
    ax.plot(X_activations[:n_points, 0], X_activations[:n_points, 1], X_activations[:n_points, 2],
            linewidth=0.7)
    plt.title("True Activations")
    ax = fig.add_subplot(222, projection=Axes3D.name)
    ax.plot(X_reconstruction[:n_points, 0], X_reconstruction[:n_points, 1], X_reconstruction[:n_points, 2],
            linewidth=0.7)
    plt.title("Latent Space")
    ax =fig.add_subplot(223, projection=Axes3D.name)
    ax.plot(X_act_simulation[:n_points, 0], X_act_simulation[:n_points, 1], X_act_simulation[:n_points, 2],
            linewidth=0.7)
    plt.title("Simulated Dynamics")
    ax =fig.add_subplot(224, projection=Axes3D.name)
    ax.plot(X_rec_simulation[:n_points, 0], X_rec_simulation[:n_points, 1], X_rec_simulation[:n_points, 2],
            linewidth=0.7)
    plt.title("Simulated Latent Dynamics")
    plt.show()


if __name__ == "__main__":
    # os.chdir("../../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1607352660  # cartpole-v1
    agent_id = 1607352660  # inverted pendulum no vel, continuous action

    parser = argparse.ArgumentParser(description="Train SindyControlAutoencoder for some RL task")

    parser.add_argument("agent_id", type=int, default=agent_id, help="Some Agent ID for a trained agent")
    parser.add_argument("--n_networks", type=int, default=1, help="Number of different networks for comparison")

    args = parser.parse_args()
