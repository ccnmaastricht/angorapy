from jax import vmap, random, jit, value_and_grad, grad
import jax.numpy as jnp
from jax.experimental import optimizers
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

    sindy_coefficients = random.uniform(key, (20, layer_sizes[-1]))
    coefficient_mask = jnp.ones((20, layer_sizes[-1]))
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


def vmap_encoding_pass(params, x):
    return vmap(encoding_pass, in_axes=(None, 0), out_axes=0)


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

    Theta = sindy_library_jax(z, 3, 3)
    sindy_predict = jnp.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
    dx_decode = z_derivative_decode(params['decoder'], z, sindy_predict)

    return [x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict]


def sindy_library_jax(z, latent_dim, poly_order):

    #library = [jnp.ones(z.shape[0])]
    library = []

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



layer_sizes = [128, 64, 32, 3]
key = random.PRNGKey(1)

input_data = random.normal(key, (10000, 128))
dx_data = random.normal(key, (10000, 128))
autoencoder, coefficient_mask = build_sindy_autoencoder(layer_sizes, key)

output = vmap(autoencoder_pass, in_axes=({'encoder': None, 'decoder': None, 'sindy_coefficients': None}, None, 0, 0))


def loss(sindy_autoencoder, coefficient_mask, input, dx_input):

    x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict = output(sindy_autoencoder, coefficient_mask, input, dx_input)

    reconstruction_loss = jnp.mean((x - x_decode)**2)
    sindy_z = jnp.mean((dz-sindy_predict)**2)
    sindy_x = jnp.mean((dx - dx_decode)**2)

    total_loss = reconstruction_loss + 0 * sindy_z + 1e-4 * sindy_x
    return total_loss


jit_loss = jit(loss)
gradient_fun = jit(grad(loss))

loss_atm = jit_loss(autoencoder, coefficient_mask, input_data, dx_data)

grads = gradient_fun(autoencoder, coefficient_mask, input_data, dx_data)


def update(params, coefficient_mask, input, dx_input, opt_state):

    value, grads = value_and_grad(loss)(params, coefficient_mask, input, dx_input)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value




step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(autoencoder)
params = get_params(opt_state)

train_loss = []

noise_strength = 1e-6
training_data = get_lorenz_data(1024, noise_strength=noise_strength)
validation_data = get_lorenz_data(20, noise_strength=noise_strength)

batch_size = 8000
num_batches = int(jnp.ceil(len(training_data['x']) / batch_size))


def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)


for epoch in range(1000):
    start = time.time()
    for i in range(num_batches):
        idx = batch_indices(i)

        params, opt_state, value = update(params, coefficient_mask,
                                        training_data['x'][idx, :],
                                        training_data['dx'][idx, :],
                                        opt_state)

        train_loss.append(value)
    stop = time.time()
    print("Epoch", str(epoch), "Loss:", str(value), "This took:", str(jnp.round(stop-start, 6)), "s")

