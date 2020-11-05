import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from analysis.sindy.lorenz_example import get_lorenz_data
from scipy.special import binom


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_layer(params, x):
    return sigmoid(np.matmul(x, params[0]) + params[1])


def build_encoder(layer_sizes, scale=1e-2):
    encoding_params = []
    for m, n in zip(layer_sizes[1:], layer_sizes[:-1]):
        w, b = scale * np.random.randn(n, m), scale * np.random.randn(m)
        encoding_params.append([w, b])
    return encoding_params


def build_decoder(layer_sizes, scale=1e-2):
    decoding_params = []
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):

        w, b = scale * np.random.randn(n, m), scale * np.random.randn(m)
        decoding_params.append([w, b])
    return decoding_params


def build_sindy_autoencoder(layer_sizes, library_size):

    encoding_params = build_encoder(layer_sizes)
    decoding_params = build_decoder(layer_sizes)

    sindy_coefficients = np.random.randn(library_size, layer_sizes[-1])
    coefficient_mask = np.ones((library_size, layer_sizes[-1]))
    params = {'encoder': encoding_params,
              'decoder': decoding_params,
              'sindy_coefficients': sindy_coefficients}
    return params, coefficient_mask


def z_derivative(params, x, dx):
    dz = dx

    for w, b in params[:-1]:
        x = np.matmul(x, w) + b
        x = sigmoid(x)
        dz = np.multiply(np.multiply(x, 1-x), np.matmul(dz, w))

    return np.matmul(dz, params[-1][0])


def z_derivative_decode(params, z, sindy_predict):
    dx_decode = sindy_predict
    params = params[::-1]

    for w, b in params[:-1]:
        z = np.matmul(z, w) + b
        z = sigmoid(z)
        dx_decode = np.multiply(np.multiply(z, 1-z), np.matmul(dx_decode, w))

    return np.matmul(dx_decode, params[-1][0])


def encoding_pass(params, x):
    activation = x

    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return np.matmul(activation, params[-1][0]) + params[-1][1]


def decoding_pass(params, input):
    activation = input
    params = params[::-1]  # reverse order for decoder
    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return np.matmul(activation, params[-1][0]) + params[-1][1]


def autoencoder_pass(params, coefficient_mask, x, dx):

    z = encoding_pass(params['encoder'], x)
    dz = z_derivative(params['encoder'], x, dx)
    x_decode = decoding_pass(params['decoder'], z)

    Theta = sindy_library_jax(z, 3, 3)
    sindy_predict = np.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
    dx_decode = z_derivative_decode(params['decoder'], z, sindy_predict)

    return [x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict]


def sindy_library_jax(z, latent_dim, poly_order):

    library = [np.ones((z.shape[0],))]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(np.multiply(z[:, i], z[:, j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(np.multiply(z[:, i], z[:, j]) * z[:, k])

    return np.stack(library, axis=1)


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


if __name__ == "__main__":
    layer_sizes = [128, 64, 32, 3]
    batch_size = 2048
    epochs = 250

    lib_size = library_size(3, 3)
    sindy_autoencoder, coefficient_mask = build_sindy_autoencoder(layer_sizes, lib_size)

    noise_strength = 1e-6
    training_data = get_lorenz_data(1024, noise_strength=noise_strength)
    validation_data = get_lorenz_data(20, noise_strength=noise_strength)

    num_batches = int(np.ceil(len(training_data['x']) / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)


    def loss(sindy_autoencoder, iter):
        idx = batch_indices(iter)
        x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict = autoencoder_pass(sindy_autoencoder,
                                                                                   coefficient_mask,
                                                                                   training_data['x'][idx, :],
                                                                                   training_data['dx'][idx, :])
        reconstruction_loss = np.mean((x - x_decode) ** 2)
        sindy_z = np.mean((dz - sindy_predict) ** 2)
        sindy_x = np.mean((dx - dx_decode) ** 2)

        total_loss = reconstruction_loss + 0 * sindy_z + 1e-4 * sindy_x
        return total_loss


    grad_fun = grad(loss)

    print("     Epoch     |    Train accuracy  ")
    def print_perf(params, iter , g):
        if iter % num_batches == 0:

            print("{:15}|{:20}".format(iter//num_batches, loss(params, iter)))

    optimized_params = sindy_autoencoder
    for i in range(5):
        optimized_params = adam(grad_fun, optimized_params, num_iters=50*num_batches, callback=print_perf,
                                step_size=1e-3)
        coefficient_mask = np.abs(optimized_params['sindy_coefficients']) > 0.1
        print("Reset coefficients")
        def loss(sindy_autoencoder, iter):
            idx = batch_indices(iter)
            x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict = autoencoder_pass(sindy_autoencoder,
                                                                                       coefficient_mask,
                                                                                       training_data['x'][idx, :],
                                                                                       training_data['dx'][idx, :])
            reconstruction_loss = np.mean((x - x_decode) ** 2)
            sindy_z = np.mean((dz - sindy_predict) ** 2)
            sindy_x = np.mean((dx - dx_decode) ** 2)

            total_loss = reconstruction_loss + 0 * sindy_z + 1e-4 * sindy_x
            return total_loss