import os
import numpy as np
import jax.numpy as jnp
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append("/home/raphael/Code/dexterous-robot-hand/")
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder import control_autoencoder, utils
from jax.experimental import optimizers
from jax import random
import sklearn.decomposition as skld
import time
import pickle
import argparse
# defaults
hps = {'system_loss_coeff': 1,
       'control_loss_coeff': 1,
       'dx_loss_weight': 1e-4,
       'dz_loss_weight': 1e-6,
       'reg_loss_weight': 1e-5}
FILE_DIR = "/home/raphael/Code/dexterous-robot-hand/analysis/sindy_autoencoder/files/"


def main(agent_id, settings: dict):

    chiefinv = Chiefinvestigator(agent_id)
    # get data for training and testing
    try:
        SAVE_DIR = "analysis/sindy_autoencoder/storage/" + str(agent_id) + "/"
        training_data = pickle.load(open(SAVE_DIR + "training_data.pkl", "rb"))
        testing_data = pickle.load(open(SAVE_DIR + "testing_data.pkl", "rb"))
        activations_all_episodes, inputs_all_episodes, actions_all_episodes, states_all_episodes, _ \
            = chiefinv.get_data_over_episodes(1, "policy_recurrent_layer", chiefinv.get_layer_names()[1])
    except FileNotFoundError:
        # collect data from episodes
        print("GENERATING DATA...")
        activations_all_episodes, inputs_all_episodes, actions_all_episodes, states_all_episodes, _ \
            = chiefinv.get_data_over_episodes(settings['n_episodes'], "policy_recurrent_layer", chiefinv.get_layer_names()[1])
        print(f"SIMULATED {settings['n_episodes']} episodes")

        # create training and testing datasets
        training_size = int(len(inputs_all_episodes) * 0.8)
        dx = np.gradient(activations_all_episodes, axis=0)
        training_data = {'x': activations_all_episodes[:training_size, :],
                         'dx': dx[:training_size, :],
                         'u': inputs_all_episodes[:training_size, :]}
        testing_data = {'x': activations_all_episodes[training_size:, :],
                        'dx': dx[training_size:, :],
                        'u': inputs_all_episodes[training_size:, :]}  # also automatically safe if generated for 1st tim
    utils.plot_training_data(states_all_episodes, FILE_DIR) # TODO: needs to be generalized

    try:
        # load trained system
        state = control_autoencoder.load_state(str(agent_id))
        params, coefficient_mask = state['autoencoder'], state['coefficient_mask']
        control_autoencoder.plot_params(params, coefficient_mask)  # TODO: extend this function to all params
        plt.savefig(FILE_DIR + "figures/" + str(agent_id) + "_params.png", dpi=300)
    except FileNotFoundError:
        lib_size = utils.library_size(settings['layers'][-1], settings['poly_order'], include_control=True)
        key = random.PRNGKey(123)

        num_batches = int(jnp.ceil(len(training_data['x']) / settings['batch_size']))
        params, coefficient_mask = control_autoencoder.build_sindy_control_autoencoder(settings['layers'], lib_size, key)
        control_autoencoder.plot_params(params, coefficient_mask)
        plt.savefig(FILE_DIR + "figures/" + str(agent_id) + "_params.png", dpi=300)

        # set up optimizer
        batch_size = settings['batch_size']
        opt_init, opt_update, get_params = optimizers.adam(settings['learning_rate'])
        opt_state = opt_init(params)

        # train
        all_train_losses = []
        start_time = time.time()
        for epoch in range(settings['epochs']):
            for batch in range(num_batches):
                ids = utils.batch_indices(batch, num_batches, batch_size)
                opt_state = control_autoencoder.update_jit(batch, opt_state, opt_update, get_params,
                                                           training_data['x'][ids, :],
                                                           training_data['dx'][ids, :],
                                                           training_data['u'][ids, :], coefficient_mask, hps)

            params = get_params(opt_state)
            if epoch % settings['thresholding_frequency'] == 0 and epoch > 1:
                coefficient_mask = jnp.abs(params['sindy_coefficients']) > settings['thresholding_coefficient']
                print("Updated coefficient mask")

            all_train_losses.append(control_autoencoder.loss_jit(params,
                                                                 training_data['x'][:batch_size, :],
                                                                 training_data['dx'][:batch_size, :],
                                                                 training_data['u'][:batch_size, :], coefficient_mask,
                                                                 hps))
            if epoch % settings['print_every'] == 0:
                utils.print_update(all_train_losses[-1]['total'], epoch, epoch * num_batches, time.time() - start_time)
                start_time = time.time()

        print(f"FINISHING TRAINING...\n"
              f"Sparsity: {jnp.sum(coefficient_mask)} active terms")

        all_train_losses = {k: [dic[k] for dic in all_train_losses] for k in all_train_losses[0]}
        time_steps = np.linspace(0, settings['epochs'], settings['epochs'])
        utils.plot_losses(time_steps, all_train_losses, FILE_DIR)

        hps['reg_loss_weight'] = 0  # no regularization
        print('REFINEMENT...')
        all_refine_losses = []

        start_time = time.time()
        for epoch in range(settings['refinement_epochs']):
            for batch in range(num_batches):
                ids = utils.batch_indices(batch, num_batches, batch_size)
                opt_state = control_autoencoder.update_jit(batch, opt_state, opt_update, get_params,
                                                           training_data['x'][ids, :],
                                                           training_data['dx'][ids, :],
                                                           training_data['u'][ids, :], coefficient_mask, hps)

            all_refine_losses.append(control_autoencoder.loss_jit(get_params(opt_state),
                                                                  training_data['x'][:batch_size, :],
                                                                  training_data['dx'][:batch_size, :],
                                                                  training_data['u'][:batch_size, :], coefficient_mask,
                                                                  hps))
            if epoch % settings['print_every'] == 0:
                utils.print_update(all_refine_losses[-1]['total'], epoch, epoch * num_batches, time.time() - start_time)
                start_time = time.time()

        all_refine_losses = {k: [dic[k] for dic in all_refine_losses] for k in all_refine_losses[0]}

        params = get_params(opt_state)
        test_batch_size = 10000  # 10 episodes
        test_loss = control_autoencoder.loss_jit(params,
                                                 testing_data['x'][:test_batch_size, :],
                                                 testing_data['dx'][:test_batch_size, :],
                                                 testing_data['u'][:test_batch_size, :],
                                                 coefficient_mask, hps)['total']
        print(f"Loss on large test batch: {round(test_loss, 6)}")

        state = {'autoencoder': get_params(opt_state),
                 'coefficient_mask': coefficient_mask,
                 'hps': {'layers': settings['layers'],
                         'poly_order': settings['poly_order'],
                         'library:size': lib_size,
                         'lr': settings['learning_rate'],
                         'epochs': settings['epochs'],
                         'batch_size': batch_size,
                         'thresholding_frequency': settings['thresholding_frequency'],
                         'threshold_coefficient': settings['thresholding_coefficient']},
                 'history': {'train_loss': all_train_losses,
                             'refinement_loss': all_refine_losses}}
        control_autoencoder.save_state(state, str(agent_id))
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
    from sympy import symbols
    theta_syms = symbols(ylabels)
    dz_syms = symbols(latex_labels)

    expr = np.matmul(theta_syms, coefficient_mask * params['sindy_coefficients'])

    plt.figure()
    for i, dz_sym in enumerate(dz_syms):
        plt.text(0.2, 1 - 0.1 * i, f"{dz_sym} = {expr[i]}")
    plt.axis('off')
    plt.savefig(FILE_DIR + "figures/" + str(agent_id) + "_sindy_equations.png", dpi=400)

    # Simulate
    n_points = 1000 # 3 episodes
    [z, y, sindy_predict] = control_autoencoder.batch_compute_latent_space(params, coefficient_mask,
                                                                           activations_all_episodes, inputs_all_episodes)
    # Simulate Dynamics and produce 3 episodes
    #_, _, simulated_activations, simulation_results, actions = control_autoencoder.simulate_episode(chiefinv,
     #                                                                                               params, coefficient_mask,
       #                                                                                             render=True)
    # Reduce Dimensions
    activation_pca = skld.PCA(3)
    X_activations = activation_pca.fit_transform(activations_all_episodes)
    reconstruction_pca = skld.PCA(3)
    X_reconstruction = reconstruction_pca.fit_transform(z)
    #X_rec_simulation = reconstruction_pca.transform(simulation_results)
    #X_act_simulation = activation_pca.transform(simulated_activations)

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
    #ax.plot(X_act_simulation[:n_points, 0], X_act_simulation[:n_points, 1], X_act_simulation[:n_points, 2],
    #        linewidth=0.7)
    #plt.title("Simulated Dynamics")
    #ax =fig.add_subplot(224, projection=Axes3D.name)
    #ax.plot(X_rec_simulation[:n_points, 0], X_rec_simulation[:n_points, 1], X_rec_simulation[:n_points, 2],
    #        linewidth=0.7)
    #plt.title("Simulated Latent Dynamics")
    plt.savefig(FILE_DIR + "figures/" + str(agent_id) + "_sim_res.png", dpi=300)


if __name__ == "__main__":
    # os.chdir("../../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # agent_id = 1607352660  # cartpole-v1
    agent_id = 1607352660  # inverted pendulum no vel, continuous action

    parser = argparse.ArgumentParser(description="Train SindyControlAutoencoder for some RL task")

    parser.add_argument("--agent_id", type=int, default=agent_id, help="Some Agent ID for a trained agent")
    parser.add_argument("--n_networks", type=int, default=1, help="Number of different networks for comparison")
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

    args = parser.parse_args()

    main(args.agent_id, vars(args))
