from jax import random, vmap
from jax.nn.initializers import xavier_uniform
from analysis.sindy_autoencoder.models.layers import *
from analysis.sindy_autoencoder.utils import sindy_library_jax


def build_encoder(layer_sizes, key, scale=1e-2, initializer: str = 'xavier_uniform'):
    keys = random.split(key, len(layer_sizes))
    encoding_params = []
    for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w_key, b_key = random.split(k)
        if initializer == 'normal':
            w, b = scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        elif initializer == 'xavier_uniform':
            w_init = xavier_uniform()
            w, b = w_init(w_key, (n, m)), scale * random.normal(b_key, (n,))
        else:
            raise ValueError(f"Weight Initializer was {initializer} which is not supported.")
        encoding_params.append([w, b])
    return encoding_params


def build_decoder(layer_sizes, key, scale=1e-2, initializer: str = 'xavier_uniform'):
    keys = random.split(key, len(layer_sizes))
    decoding_params = []
    for m, n, k in zip(layer_sizes[1:], layer_sizes[:-1], keys):
        w_key, b_key = random.split(k)
        if initializer == 'normal':
            w, b = scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        elif initializer == 'xavier_uniform':
            w_init = xavier_uniform()
            w, b = w_init(w_key, (n, m)), scale * random.normal(b_key, (n,))  # SindyAutoencoder
        else:
            raise ValueError(f"Weight Initializer was {initializer} which is not supported.")
        decoding_params.append([w, b])
    return decoding_params


def build_sindy_autoencoder(layer_sizes, library_size, key, initializer: str = 'xavier_uniform',
                            sindy_initializer: str = 'constant'):
    key = random.split(key, 3)
    key = (k for k in key)

    encoding_params = build_encoder(layer_sizes, next(key), initializer=initializer)
    decoding_params = build_decoder(layer_sizes, next(key), initializer=initializer)

    if sindy_initializer == 'constant':
        sindy_coefficients = jnp.ones((library_size, layer_sizes[-1]))
    elif sindy_initializer == 'xavier_uniform':
        sindy_init = xavier_uniform()
        sindy_coefficients = sindy_init(next(key), (library_size, layer_sizes[-1]))
    elif sindy_initializer == 'normal':
        sindy_coefficients = random.normal(next(key), (library_size, layer_sizes[-1]))
    else:
        raise ValueError(f"Sindy Initializer was {sindy_initializer} which is not supported. Try to specify one of \n"
                         f"constant, xavier_uniform or normal")

    coefficient_mask = jnp.ones((library_size, layer_sizes[-1]))

    params = {'encoder': encoding_params,
              'decoder': decoding_params,
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


def z_derivative(params, x, dx):
    dz = dx

    for w, b in params[:-1]:
        x = jnp.matmul(w, x) + b
        x = sigmoid(x)
        dz = jnp.multiply(jnp.multiply(x, 1 - x), jnp.matmul(w, dz))

    return jnp.matmul(params[-1][0], dz)


def z_derivative_decode(params, z, sindy_predict):
    dx_decode = sindy_predict
    params = params[::-1]

    for w, b in params[:-1]:
        z = jnp.matmul(w, z) + b
        z = sigmoid(z)
        dx_decode = jnp.multiply(jnp.multiply(z, 1 - z), jnp.matmul(w, dx_decode))

    return jnp.matmul(params[-1][0], dx_decode)


def autoencoder_pass(params, coefficient_mask, x, dx, u, lib_size,
                     poly_order: int = 2, include_sine: bool = False):

    z = encoding_pass(params['encoder'], x)
    x_decode = decoding_pass(params['decoder'], z)
    dz = z_derivative(params['encoder'], x, dx)

    c = jnp.concatenate((z, u))
    Theta = sindy_library_jax(c, lib_size, poly_order=poly_order, include_sine=include_sine)
    sindy_predict = jnp.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
    dx_decode = z_derivative_decode(params['decoder'], z, sindy_predict)

    return [x_decode, dz, sindy_predict, dx_decode]


batch_autoencoder = vmap(autoencoder_pass, in_axes=({'encoder': None,
                                                     'decoder': None,
                                                     'sindy_coefficients': None},
                                                    None,
                                                    0, 0, 0,
                                                    None, None, None))
