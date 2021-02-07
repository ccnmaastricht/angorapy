import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append("/home/raphael/Code/dexterous-robot-hand/")
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder import control_autoencoder, utils
import sklearn.decomposition as skld
import pickle
import argparse
from sympy import symbols
# defaults
hps = {'system_loss_coeff': 1,
       'control_loss_coeff': 1,
       'dx_loss_weight': 1e-4,
       'dz_loss_weight': 1e-6,
       'reg_loss_weight': 1e-5}


def main(agent_id, settings: dict):

    chiefinv = Chiefinvestigator(agent_id)
    # get data for training and testing
    FILE_DIR = "/home/raphael/Code/dexterous-robot-hand/analysis/sindy_autoencoder/files/" + str(agent_id) + "/"
    SAVE_DIR = "analysis/sindy_autoencoder/storage/" + str(agent_id) + "/"
    try:
        training_data = pickle.load(open(SAVE_DIR + "training_data.pkl", "rb"))
        testing_data = pickle.load(open(SAVE_DIR + "testing_data.pkl", "rb"))
        activations_all_episodes, inputs_all_episodes, actions_all_episodes, states_all_episodes, _ \
            = chiefinv.get_data_over_episodes(1, "policy_recurrent_layer", chiefinv.get_layer_names()[1])
        episode_size = len(states_all_episodes)
    except FileNotFoundError:
        training_data, testing_data, states_all_episodes, episode_size = chiefinv.get_sindy_datasets(settings)
        utils.save_data(training_data, testing_data, SAVE_DIR)
    utils.plot_training_data(states_all_episodes, episode_size, FILE_DIR) # TODO: needs to be generalized

    try:
        # load trained system
        state = control_autoencoder.load_state(str(agent_id))
        params, coefficient_mask = state['autoencoder'], state['coefficient_mask']
        control_autoencoder.plot_params(params, coefficient_mask)  # TODO: extend this function to all params
        plt.savefig(FILE_DIR + "figures/" + "params.png", dpi=300)
    except FileNotFoundError:
        state = control_autoencoder.train(training_data, testing_data, settings, hps, FILE_DIR)
        params, coefficient_mask = state['autoencoder'], state['coefficient_mask']

    # plot sindy coefficients
    xlabels, ylabels, latex_labels = utils.generate_labels(settings['layers'][-1], settings['poly_order'])
    plt.figure(figsize=(10, 20))
    plt.spy(coefficient_mask * params['sindy_coefficients'],
            marker='o', markersize=10, aspect='auto')
    plt.xticks([0, 1, 2, 3], latex_labels, size=12)
    yticks = list(np.arange(len(coefficient_mask)))
    plt.yticks(yticks, ylabels, size=12)
    plt.savefig(FILE_DIR + "figures/" + str(agent_id) + "_sindy_coefficients.png", dpi=400)

    # Print Sparse State Equations
    theta_syms = symbols(ylabels)
    dz_syms = symbols(latex_labels)
    expr = np.matmul(theta_syms, coefficient_mask * params['sindy_coefficients'])

    plt.figure()
    for i, dz_sym in enumerate(dz_syms):
        plt.text(0.2, 1 - 0.1 * i, f"{dz_sym} = {expr[i]}")
    plt.axis('off')
    plt.savefig(FILE_DIR + "figures/" + str(agent_id) + "_sindy_equations.png", dpi=400)

    # Simulate
    n_points = 1000  # 3 episodes
    z, _, _ = control_autoencoder.batch_compute_latent_space(params, coefficient_mask,
                                                             training_data['x'], training_data['u'])
    # test
    z_test, _, _ = control_autoencoder.batch_compute_latent_space(params, coefficient_mask,
                                                                  testing_data['x'], testing_data['u'])
    beta = utils.regress(training_data['a'], z, l=1)
    params['output_weights'] = beta
    a_pred = np.matmul(z_test, beta)
    print(f"Error for action prediction from latent space {np.mean(np.square((a_pred - testing_data['a'])))}")
    # Simulate Dynamics and produce 3 episodes
    _, _, simulated_activations, simulation_results, actions = control_autoencoder.simulate_episode(chiefinv,
                                                                                                    params,
                                                                                                    coefficient_mask,
                                                                                                    render=False)
    # Reduce Dimensions
    activation_pca = skld.PCA(3)
    X_activations = activation_pca.fit_transform(training_data['x'])
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
    plt.savefig(FILE_DIR + "figures/" + str(agent_id) + "_sim_res.png", dpi=300)


if __name__ == "__main__":
    # os.chdir("../../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1607352660  # cartpole-v1
    agent_id = 1607352660  # inverted pendulum no vel, continuous action

    parser = argparse.ArgumentParser(description="Train SindyControlAutoencoder for some RL task")

    parser.add_argument("agent_id", type=int, help="Some Agent ID for a trained agent")
    parser.add_argument("--n_networks", type=int, default=1, help="Number of different networks for comparison")
    parser.add_argument("--seed", type=int, default=123, help="Seed for key generation")
    parser.add_argument("--poly_order", type=int, default=2, help="Polynomial Order in Sindy Library")
    parser.add_argument("--layers", type=list, default=[64, 32, 8, 4], help="List of layer sizes for autoencoder")
    parser.add_argument("--thresholding_frequency", type=int, default=500,
                        help="Number of epochs after which the coefficient mask will be updated")
    parser.add_argument("--thresholding_coefficient", type=float, default=0.1,
                        help="Thresholding coefficient below which coefficients will be set to zero.")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch Size for training and refinement")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Step size for updating parameters by")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs to train for")
    parser.add_argument("--refine_epochs", type=int, default=1000, help="Number of refinement epochs")
    parser.add_argument("--print_every", type=int, default=200, help="Number of epochs at which to print an update")

    parser.add_argument("--n_episodes", type=int, default=500, help="Number of episodes to generate training and "
                                                                    "testing data from.")

    args = parser.parse_args()

    main(args.agent_id, vars(args))
