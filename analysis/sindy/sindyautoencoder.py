import jax.numpy as jnp
from jax import grad, vmap, random, value_and_grad, jit
from jax.experimental import optimizers
from scipy.integrate import odeint
import time
from analysis.sindy.lorenz_example import get_lorenz_data


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


def build_sindy_autoencoder(layer_sizes, key):

    encoding_params = build_encoder(layer_sizes, key)
    decoding_params = build_decoder(layer_sizes, key)

    sindy_coefficients = random.uniform(key, (12, layer_sizes[-1]))
    coefficient_mask = jnp.ones((12, layer_sizes[-1]))
    params = {'encoder': encoding_params,
              'decoder': decoding_params,
              'sindy_coefficients': sindy_coefficients}
    return params, coefficient_mask


@jit
def z_derivative(params, x, dx):
    dz = dx

    for w, b in params[:-1]:
        x = jnp.matmul(w, x) + b
        x = sigmoid(x)
        dz = jnp.multiply(jnp.multiply(x, 1-x), jnp.matmul(w, dz))

    return jnp.matmul(params[-1][0], dz)


@jit
def z_derivative_decode(params, z, sindy_predict):
    dx_decode = sindy_predict
    params = params[::-1]

    for w, b in params[:-1]:
        z = jnp.matmul(w, z) + b
        z = sigmoid(z)
        dx_decode = jnp.multiply(jnp.multiply(z, 1-z), jnp.matmul(w, dx_decode))

    return jnp.matmul(params[-1][0], dx_decode)


@jit
def encoding_pass(params, x):
    activation = x

    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return jnp.matmul(params[-1][0], activation) + params[-1][1]


@jit
def decoding_pass(params, input):
    activation = input
    params = params[::-1]  # reverse order for decoder
    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return jnp.matmul(params[-1][0], activation) + params[-1][1]


@jit
def autoencoder_pass(params, coefficient_mask, x, dx):

    z = encoding_pass(params['encoder'], x)
    dz = z_derivative(params['encoder'], x, dx)
    x_decode = decoding_pass(params['decoder'], z)

    Theta = sindy_library_jax(z, 3, 2)
    sindy_predict = jnp.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
    dx_decode = z_derivative_decode(params['decoder'], z, sindy_predict)

    return [x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict]


def sindy_library_jax(z, latent_dim, poly_order):

    library = [jnp.ones(z.shape[0])]

    for i in range(latent_dim):
        library.append(1)

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


def loss(sindy_autoencoder, coefficient_mask, input, dx_input):

    activations = zip(*[autoencoder_pass(sindy_autoencoder, coefficient_mask, input[i, :], dx_input[i, :]) for i in range(input.shape[0])])
    x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict = map(jnp.stack, activations)

    reconstruction_loss = jnp.mean((x - x_decode)**2)
    sindy_z = jnp.mean((dz-sindy_predict)**2)
    sindy_x = jnp.mean((dx - dx_decode)**2)

    total_loss = reconstruction_loss + 0 * sindy_z + 1e-4 * sindy_x
    return total_loss


def update(params, coefficient_mask, input, dx_input, opt_state):

    value, grads = value_and_grad(loss)(params, coefficient_mask, input, dx_input)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


if __name__ == "__main__":
    layer_sizes = [128, 64, 32, 3]
    key = random.PRNGKey(1)
    num_epochs = 5001
    coefficient_threshold = 0.1
    threshold_frequency = 500

    sindy_autoencoder, coefficient_mask = build_sindy_autoencoder(layer_sizes, key)



    #total_loss = loss(sindy_autoencoder, input, dx_input)
    #print(total_loss)

    step_size = 1e-3
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(sindy_autoencoder)
    params = get_params(opt_state)

    train_loss = []

    noise_strength = 1e-6
    training_data = get_lorenz_data(1024, noise_strength=noise_strength)
    validation_data = get_lorenz_data(20, noise_strength=noise_strength)


    for i in range(int(num_epochs)):
        for k in range(250):
            sidx = k*1024
            eidx = k * 1024 +1024
            params, opt_state, los = update(params, coefficient_mask,
                                            training_data['x'][sidx:eidx, :],
                                            training_data['dx'][sidx:eidx, :],
                                            opt_state)
            train_loss.append(los)
            if k % threshold_frequency == 0:

                 coefficient_mask = jnp.float32(jnp.abs(params['sindy_coefficients']) > coefficient_threshold)

            print(los)

