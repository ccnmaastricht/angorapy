import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder import control_autoencoder, utils
from jax.experimental import optimizers
from jax import random
import sklearn.decomposition as skld
import time
import pickle

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# agent_id = 1607352660  # cartpole-v1
agent_id = 1607352660 # inverted pendulum no vel, continuous action

chiefinv = Chiefinvestigator(agent_id)

SAVE_DIR = "analysis/sindy_autoencoder/storage/"
training_data = pickle.load(open(SAVE_DIR + "training_data.pkl", "rb"))
testing_data = pickle.load(open(SAVE_DIR + "testing_data.pkl", "rb"))
n_episodes = 1
activations_all_episodes, inputs_all_episodes, actions_all_episodes, states_all_episodes, _ \
    = chiefinv.get_data_over_episodes(n_episodes, "policy_recurrent_layer", chiefinv.get_layer_names()[1])

layers = [64, 32, 8, 4]
poly_order = 2
lib_size = utils.library_size(layers[-1], poly_order, include_control=True)
key = random.PRNGKey(123)

thresholding_frequency, coefficient_theshold = 500, 0.1
hps = {'system_loss_coeff': 1,
       'control_loss_coeff': 1,
       'dx_loss_weight': 1e-4,
       'dz_loss_weight': 1e-6,
       'reg_loss_weight': 1e-5}

batch_size = 5000
num_batches = int(jnp.ceil(len(training_data['x']) / batch_size))
init_params, coefficient_mask = control_autoencoder.build_sindy_control_autoencoder(layers, lib_size, key)

control_autoencoder.plot_params(init_params, coefficient_mask)

state = control_autoencoder.load_state("SindyControlAutoencoder")
params, coefficient_mask = state['autoencoder'], state['coefficient_mask']

# Simulate
n_points = 1000 # 3 episodes
[z, y, sindy_predict] = control_autoencoder.batch_compute_latent_space(params, coefficient_mask,
                                                                       activations_all_episodes, inputs_all_episodes)
# Simulate Dynamics and produce 3 episodes
_, _, simulated_activations, simulation_results, actions = control_autoencoder.simulate_episode(chiefinv,
                                                                                                params, coefficient_mask,
                                                                                                render=False)
# Reduce Dimensions
activation_pca = skld.PCA(3)
X_activations = activation_pca.fit_transform(activations_all_episodes)
reconstruction_pca = skld.PCA(3)
X_reconstruction = reconstruction_pca.fit_transform(z)
X_rec_simulation = reconstruction_pca.transform(simulation_results)
X_act_simulation = activation_pca.transform(simulated_activations)

fig = plt.figure()
ax = fig.add_subplot(131, projection=Axes3D.name)
ax.plot(X_activations[:n_points, 0], X_activations[:n_points, 1], X_activations[:n_points, 2],
        linewidth=0.7)
plt.title("True Activations")
ax = fig.add_subplot(132, projection=Axes3D.name)
ax.plot(X_reconstruction[:n_points, 0], X_reconstruction[:n_points, 1], X_reconstruction[:n_points, 2],
        linewidth=0.7)
plt.title("Latent Space")
ax =fig.add_subplot(133, projection=Axes3D.name)
ax.plot(X_act_simulation[:n_points, 0], X_act_simulation[:n_points, 1], X_act_simulation[:n_points, 2],
        linewidth=0.7)
plt.title("Simulated Dynamics")
plt.show()
