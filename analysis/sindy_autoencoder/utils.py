import jax.numpy as jnp
from scipy.special import binom


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


def generate_labels(zlabels, ulabels, poly_order):

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

    return zlabels, all_labels


def batch_indices(iter, num_batches, batch_size):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx + 1) * batch_size)


def print_update(train_loss, epoch, n_updates, dt):
    print(f"Epoch {epoch}",
          f"| Loss {round(train_loss, 7)}",
          f"| Updates {n_updates}",
          f"| This took: {round(dt, 4)}s")

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