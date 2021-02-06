import jax.numpy as jnp
from scipy.special import binom
import matplotlib.pyplot as plt
import numpy as np
import os


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
        zlabels.append('z'+str(i))
        ulabels.append('y'+str(i))
        latex_labels.append(r"$\dot{z}_"+str(i)+"$")

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


def plot_training_data(states_all_episodes, FILE_DIR):  # needs generalization towards task
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, 1000 * 0.02, 1000)
    plt.subplot(121)
    plt.plot(x, states_all_episodes[:1000, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Cart Position")

    plt.subplot(122)
    plt.plot(x, states_all_episodes[:1000, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Pole Position")
    plt.savefig(FILE_DIR + "figures/InvPend.png", dpi=300)


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
    plt.savefig(FILE_DIR + "figures/losses.png", dpi=300)
'''
def regress(Y, X, l=0.):


Parameters
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
    
'''