import os
import sys
sys.path.append("/home/raphael/Code/dexterous-robot-hand/")
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder import control_autoencoder, utils
import pickle
import argparse
# defaults
hps = {'system_loss_coeff': 1,
       'control_loss_coeff': 1,
       'dx_loss_weight': 0.5,
       'dz_loss_weight': 0.01,
       'dy_loss_weight': 0.01,
       'du_loss_weight': 0.5,
       'reg_loss_weight': 0.1,
       'reg_u_loss_weight': 0.1}


def main(agent_id, settings: dict, training_data=None, testing_data=None):
    utils.print_behaviour(settings)
    FILE_DIR = "/home/raphael/Code/dexterous-robot-hand/analysis/sindy_autoencoder/files/" + str(agent_id) + "/"
    SAVE_DIR = "analysis/sindy_autoencoder/storage/" + str(agent_id) + "/"


    try:
        # load trained system
        state = control_autoencoder.load_state(str(agent_id))
        params, coefficient_mask = state['autoencoder'], state['coefficient_mask']
    except FileNotFoundError:
        state = control_autoencoder.train(training_data, testing_data, settings, hps, FILE_DIR)
        params, coefficient_mask = state['autoencoder'], state['coefficient_mask']
    control_autoencoder.plot_params(params, coefficient_mask, FILE_DIR)  # TODO: extend this function to all params

    utils.plot_coefficients(params, coefficient_mask[0], settings, FILE_DIR)
    utils.plot_equations(params, coefficient_mask[0], settings, FILE_DIR)

    # Simulate
    n_points = 1000  # 3 episodes
    #z, _, _ = control_autoencoder.batch_compute_latent_space(params, coefficient_mask,
    #                                                         training_data['x'], training_data['u'])
    # Simulate Dynamics and produce 3 episodes
    #_, _, simulated_activations, simulation_results, actions = control_autoencoder.simulate_episode(chiefinv,
    #                                                                                                params,
    #                                                                                                coefficient_mask,
    #                                                                                                render=False)
    #utils.plot_simulations(training_data, simulation_results, simulated_activations, z, n_points,
    #                       FILE_DIR)


if __name__ == "__main__":
    # os.chdir("../../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # agent_id = 1607352660  # cartpole-v1
    # agent_id = 1607352660  # inverted pendulum no vel, continuous action

    chiefinv = Chiefinvestigator(agent_id)
    # get data for training and testing
    FILE_DIR = "/home/raphael/Code/dexterous-robot-hand/analysis/sindy_autoencoder/files/" + str(agent_id) + "/"
    SAVE_DIR = "analysis/sindy_autoencoder/storage/" + str(agent_id) + "/"

    try:
        training_data = pickle.load(open(SAVE_DIR + "training_data.pkl", "rb"))
        testing_data = pickle.load(open(SAVE_DIR + "testing_data.pkl", "rb"))
        print("LOADING DATA...")
    except FileNotFoundError:
        training_data, testing_data = chiefinv.get_sindy_datasets(settings)
        utils.save_data(training_data, testing_data, SAVE_DIR)
    utils.plot_training_data(training_data, FILE_DIR)  # TODO: needs to be generalized to tasks

    parser = argparse.ArgumentParser(description="Train SindyControlAutoencoder for some RL task")

    parser.add_argument("agent_id", type=int, help="Some Agent ID for a trained agent")
    parser.add_argument("--n_networks", type=int, default=1, help="Number of different networks for comparison")
    parser.add_argument("--seed", type=int, default=123, help="Seed for key generation")
    parser.add_argument("--poly_order", type=int, default=3, help="Polynomial Order in Sindy Library")
    parser.add_argument("--layers", type=list, default=[64, 32, 4], help="List of layer sizes for autoencoder")
    parser.add_argument("--thresholding_frequency", type=int, default=500,
                        help="Number of epochs after which the coefficient mask will be updated")
    parser.add_argument("--thresholding_coefficient", type=float, default=0.1,
                        help="Thresholding coefficient below which coefficients will be set to zero.")
    parser.add_argument("--batch_size", type=int, default=8000, help="Batch Size for training and refinement")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                        help="Step size for updating parameters by")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs to train for")
    parser.add_argument("--refine_epochs", type=int, default=1000, help="Number of refinement epochs")
    parser.add_argument("--print_every", type=int, default=200, help="Number of epochs at which to print an update")

    parser.add_argument("--n_episodes", type=int, default=500, help="Number of episodes to generate training and "
                                                                    "testing data from.")

    args = parser.parse_args()

    main(args.agent_id, vars(args))
