import jax.numpy as jnp
from jax import random
from scipy.special import binom


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def sigmoid_layer(params, x):
    return sigmoid(jnp.matmul(params[0], x) + params[1])


def build_encoder(layer_sizes, key, scale=1e-2):
    keys = random.split(key, len(layer_sizes))
    encoding_params = []
    for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w_key, b_key = random.split(k)
        w, b = scale * random.normal(w_key, (n, m)), 0 * random.normal(b_key, (n, ))
        encoding_params.append([w, b])
    return encoding_params


def build_decoder(layer_sizes, key, scale=1e-2):
    keys = random.split(key, len(layer_sizes))
    decoding_params = []
    for m, n, k in zip(layer_sizes[1:], layer_sizes[:-1], keys):
        w_key, b_key = random.split(k)
        w, b = scale * random.normal(w_key, (n, m)), 0 * random.normal(b_key, (n,))
        decoding_params.append([w, b])
    return decoding_params


def build_sindy_autoencoder(layer_sizes, library_size, key):

    encoding_params = build_encoder(layer_sizes, key)
    decoding_params = build_decoder(layer_sizes, key)

    sindy_coefficients = jnp.ones((library_size, layer_sizes[-1]))
    coefficient_mask = jnp.ones((library_size, layer_sizes[-1]))
    params = {'encoder': encoding_params,
              'decoder': decoding_params,
              'sindy_coefficients': sindy_coefficients}
    return params, coefficient_mask


def z_derivative(params, x, dx):
    dz = dx

    for w, b in params[:-1]:
        x = jnp.matmul(w, x) + b
        x = sigmoid(x)
        dz = jnp.multiply(jnp.multiply(x, 1-x), jnp.matmul(w, dz))

    return jnp.matmul(params[-1][0], dz)


def z_derivative_decode(params, z, sindy_predict):
    dx_decode = sindy_predict
    params = params[::-1]

    for w, b in params[:-1]:
        z = jnp.matmul(w, z) + b
        z = sigmoid(z)
        dx_decode = jnp.multiply(jnp.multiply(z, 1-z), jnp.matmul(w, dx_decode))

    return jnp.matmul(params[-1][0], dx_decode)


def encoding_pass(params, x):
    activation = x

    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return jnp.matmul(params[-1][0], activation) + params[-1][1]


def decoding_pass(params, input):
    activation = input
    params = params[::-1]  # reverse order for decoder
    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return jnp.matmul(params[-1][0], activation) + params[-1][1]


def autoencoder_pass(params, coefficient_mask, x, dx):

    z = encoding_pass(params['encoder'], x)
    dz = z_derivative(params['encoder'], x, dx)
    x_decode = decoding_pass(params['decoder'], z)

    Theta = sindy_library_jax(z, len(params['encoder'][-1][0]), 2)
    sindy_predict = jnp.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
    dx_decode = z_derivative_decode(params['decoder'], z, sindy_predict)

    return [x, dx, dz, x_decode, dx_decode, sindy_predict]


def sindy_library_jax(z, latent_dim, poly_order):

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
                    library.append(jnp.multiply(z[i], z[j]) * z[k])

    return jnp.stack(library, axis=0)


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l