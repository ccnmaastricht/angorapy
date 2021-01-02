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
        w, b = scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n, ))
        encoding_params.append([w, b])
    return encoding_params


def build_decoder(layer_sizes, key, scale=1e-2):
    keys = random.split(key, len(layer_sizes))
    decoding_params = []
    for m, n, k in zip(layer_sizes[1:], layer_sizes[:-1], keys):
        w_key, b_key = random.split(k)
        w, b = scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        decoding_params.append([w, b])
    return decoding_params


def build_sindy_control_autoencoder(layer_sizes, library_size, key):
    key = random.split(key, 4)
    encoding_params = build_encoder(layer_sizes, next(key))
    decoding_params = build_decoder(layer_sizes, next(key))

    control_encoding_params = build_encoder(layer_sizes, next(key))
    control_decoding_params = build_decoder(layer_sizes, next(key))

    sindy_coefficients = jnp.ones((library_size, 2 * layer_sizes[-1]))
    coefficient_mask = jnp.ones((library_size, 2 * layer_sizes[-1]))
    action_coefficients = jnp.ones((2 * layer_sizes[-1], ))

    params = {'encoder': encoding_params,
              'decoder': decoding_params,
              'control_encoder': control_encoding_params,
              'control_decoder': control_decoding_params,
              'action_coefficients': action_coefficients,
              'sindy_coefficients': sindy_coefficients}

    return params, coefficient_mask


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


def control_autoencoder_pass(params, coefficient_mask, x, u, a):

    z = encoding_pass(params['encoder'], x)
    x_decode = decoding_pass(params['decoder'], z)

    y = encoding_pass(params['control_encoder'], u)
    u_decode = decoding_pass(params['control_decoder'], y)

    c = jnp.concatenate((z, y))
    Theta = sindy_library_jax(c, 2 * len(params['encoder'][-1][0]), 2)
    sindy_predict = jnp.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
    a_predict = jnp.matmul(sindy_predict, params['action_coefficients'])

    return [x_decode, u_decode, a_predict, sindy_predict]




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