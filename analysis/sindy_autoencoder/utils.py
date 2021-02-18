import jax.numpy as jnp
from scipy.special import binom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
import scipy as sc
import sklearn.decomposition as skld
from sympy import symbols
from utilities.const import COLORS


def sindy_library_jax(z, latent_dim, poly_order, include_sine=False):

    library = [1]

    for i in range(latent_dim):
        library.append(z[i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(jnp.multiply(z[i], z[j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(jnp.multiply(jnp.multiply(z[i], z[j]), z[k]))

    if include_sine:
        for i in range(latent_dim):
            library.append(jnp.sin(z[i]))

    return jnp.stack(library, axis=0)


def library_size(n, poly_order, use_sine=False, include_constant=True, include_control=False):
    l = 0
    if include_control:
        n = n * 2
    for k in range(poly_order+1):
        l += int(binom(n+k-1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def generate_labels(ndim, poly_order):

    zlabels, ulabels, latex_labels = [], [], []
    for i in range(ndim):
        zlabels.append('z'+str(i+1))
        ulabels.append('y'+str(i+1))
        latex_labels.append(r"$\dot{z}_"+str(i+1)+"$")

    combined_labels = zlabels + ulabels
    latent_dim = len(combined_labels)

    all_labels = ['1']

    for i in range(latent_dim):
        all_labels.append(combined_labels[i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                all_labels.append(combined_labels[i] + combined_labels[j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    all_labels.append(combined_labels[i] + combined_labels[j] + combined_labels[k])

    return zlabels, all_labels, latex_labels


def batch_indices(iter, num_batches, batch_size):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx + 1) * batch_size)


def print_update(train_loss, epoch, n_updates, dt):
    print(f"Epoch {epoch}",
          f"| Loss {round(train_loss, 7)}",
          f"| Updates {n_updates}",
          f"| This took: {round(dt, 4)}s")


def save_data(training_data, testing_data, save_dir):
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass

    with open(file=save_dir + 'training_data.pkl', mode='wb') as f:
        pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)

    with open(file=save_dir + 'testing_data.pkl', mode='wb') as f:
        pickle.dump(testing_data, f, pickle.HIGHEST_PROTOCOL)
    print("SAVED EPISODE DATA")


def plot_training_data(training_data, FILE_DIR):  # needs generalization towards task
    states_all_episodes = training_data['s']
    episode_size = training_data['e_size']

    plt.figure(figsize=(12, 5))
    x = np.linspace(0, episode_size * 0.02, episode_size)
    plt.subplot(121)
    plt.plot(x, states_all_episodes[:episode_size, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Cart Position")

    plt.subplot(122)
    plt.plot(x, states_all_episodes[:episode_size, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Pole Position")
    try:
        os.mkdir(FILE_DIR)
        os.mkdir(FILE_DIR + "figures/")
    except FileExistsError:
        pass

    plt.savefig(FILE_DIR + "figures/Episode.png", dpi=300)


def plot_losses(time_steps, all_train_losses, FILE_DIR):
    plt.figure()
    plt.subplot(231)
    plt.plot(time_steps, all_train_losses['total'], 'k')
    plt.title('Total Loss')

    plt.subplot(232)
    plt.plot(time_steps, all_train_losses['sys_loss'], 'r', label='System')
    plt.plot(time_steps, all_train_losses['control_loss'], 'g', label='Control')
    plt.legend()
    plt.title('Reconstruction Losses')

    plt.subplot(233)
    plt.plot(time_steps, all_train_losses['sindy_z_loss'], 'b')
    plt.title('Sindy z loss')

    plt.subplot(234)
    plt.plot(time_steps, all_train_losses['sindy_x_loss'], 'g')
    plt.title('Sindy x loss')

    plt.subplot(235)
    plt.plot(time_steps, all_train_losses['sindy_regularization_loss'], 'r')
    plt.title('L2 Loss')

    plt.subplot(236)
    plt.plot(time_steps, all_train_losses['sindy_u_loss'], 'b')
    plt.title('Action Loss')

    plt.savefig(FILE_DIR + "figures/losses.png", dpi=300)


def plot_coefficients(params, coefficient_mask, settings, FILE_DIR):
    # plot sindy coefficients
    xlabels, ylabels, latex_labels = generate_labels(settings['layers'][-1], settings['poly_order'])
    plt.figure(figsize=(10, 20))
    plt.spy(coefficient_mask * params['sindy_coefficients'],
            marker='o', markersize=10, aspect='auto')
    plt.xticks([0, 1, 2, 3], latex_labels, size=12)
    yticks = list(np.arange(len(coefficient_mask)))
    plt.yticks(yticks, ylabels, size=12)
    plt.savefig(FILE_DIR + "figures/" + "sindy_coefficients.png", dpi=400)


def plot_equations(params, coefficient_mask, settings, FILE_DIR):
    # Print Sparse State Equations
    xlabels, ylabels, latex_labels = generate_labels(settings['layers'][-1], settings['poly_order'])
    theta_syms = symbols(ylabels)
    dz_syms = symbols(latex_labels)
    expr = np.matmul(theta_syms, coefficient_mask * params['sindy_coefficients'])

    plt.figure()
    for i, dz_sym in enumerate(dz_syms):
        plt.text(0.2, 1 - 0.1 * i, f"{dz_sym} = {expr[i]}")
    plt.axis('off')
    plt.savefig(FILE_DIR + "figures/" + "sindy_equations.png", dpi=400)


def plot_simulations(training_data, simulation_results, simulated_activations, z,
                     n_points, FILE_DIR):
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
    plt.savefig(FILE_DIR + "figures/" + "sim_res.png", dpi=300)


bc, ec, wn = COLORS["HEADER"], COLORS["ENDC"], COLORS["WARNING"]


def print_behaviour(settings):
    print(f"Running {bc}SindyControlAutoencoder{ec} \n"
          f"------------------------------------------ \n"
          f"{wn}Hyperparameters:{ec} {settings}\n"
          f"------------------------------------------ \n")


def regress(Y, X, l=0.):

    """Parameters
    ----------
    Y : floating point array (observations-by-outcomes)
        outcome variables
    X : floating pint array (observation-by-predictors)
        predictors
    l : float
        (optional) ridge penalty parameter

    Returns
    -------
    beta : floating point array (predictors-by-outcomes)
        beta coefficients
    """

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

