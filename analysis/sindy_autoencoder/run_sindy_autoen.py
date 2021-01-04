import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder import control_autoencoder, utils
from jax.experimental import optimizers
from jax import random
import time

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# agent_id = 1607352660  # cartpole-v1
agent_id = 1607352660 # inverted pendulum no vel, continuous action
# agent_id = 1586597938# finger tapping

chiefinvesti = Chiefinvestigator(agent_id)

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 200
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done \
    = chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])
print(f"SIMULATED {n_episodes} episodes")

dx = np.gradient(activations_over_all_episodes, axis=0)
training_data = {'x': activations_over_all_episodes,
                 'dx': dx,
                 'u': inputs_over_all_episodes}

layers = [64, 32, 8, 4]
seed = 300
thresholding_frequency = 500
lib_size = control_autoencoder.library_size(layers[-1], 2, include_control=True)
key = random.PRNGKey(seed)
hps = {'system_loss_coeff': 1,
       'control_loss_coeff': 1,
       'dx_loss_weight': 1e-4,
       'dz_loss_weight': 1e-6,
       'reg_loss_weight': 1e-5}
batch_size = 2000
num_batches = int(jnp.ceil(len(training_data['x']) / batch_size))


def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx + 1) * batch_size)


learning_rate = 1e-3
init_params, coefficient_mask = control_autoencoder.build_sindy_control_autoencoder(layers, lib_size, key)
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(init_params)

start_time = time.time()
all_train_losses = []
print_every = 100

for epoch in range(1000):
    for batch in range(num_batches):
        ids = batch_indices(batch)
        opt_state = control_autoencoder.update_jit(batch, opt_state, opt_update, get_params,
                                                   training_data['x'][ids, :],
                                                   training_data['dx'][ids, :],
                                                   training_data['u'][ids, :], coefficient_mask, hps)

    params = get_params(opt_state)
    if epoch % thresholding_frequency == 0 and epoch > 1:
        coefficient_mask = jnp.abs(params['sindy_coefficients']) > 0.1
        print("Updated coefficient mask")

    all_train_losses.append(control_autoencoder.loss_jit(params,
                                                         training_data['x'][ids, :],
                                                         training_data['dx'][ids, :],
                                                         training_data['u'][ids, :], coefficient_mask, hps))
    train_loss = all_train_losses[-1]['total']
    batch_time = time.time() - start_time
    s = "Epoch {} in {:0.2f} sec, training loss {:0.4f}"
    print(s.format(epoch+1, batch_time, train_loss))
    start_time = time.time()

# List of dicts to dict of lists
all_train_losses = {k: [dic[k] for dic in all_train_losses] for k in all_train_losses[0]}


zlabels = ['z1', 'z2', 'z3', 'z4']
ulabels = ['y1', 'y2', 'y3', 'y4']
latex_labels = [r'$\dot{z1}$', r'$\dot{z2}$', r'$\dot{z3}$', r'$\dot{z4}$']
xlabels, ylabels = utils.generate_labels(zlabels, ulabels, 2)

fig, ax = plt.subplots(figsize=(10, 20))
params = get_params(opt_state)
plt.spy(coefficient_mask * params['sindy_coefficients'],
        marker='o', markersize=10, aspect='auto')
plt.xticks([0, 1, 2, 3], latex_labels, size=12)
yticks = list(np.arange(len(coefficient_mask)))
plt.yticks(yticks, ylabels, size=12)
plt.show()