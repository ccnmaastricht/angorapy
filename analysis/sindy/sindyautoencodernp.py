import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from analysis.sindy.lorenz_example import get_lorenz_data
from scipy.special import binom
from analysis.sindy.sindy_utils import library_size, sindy_library
from analysis.sindy.autoencoder_utils import build_encoder, build_decoder, encoding_pass, \
    decoding_pass, z_derivative, z_derivative_decode


class SindyAutoencoder(object):

    def __init__(self, layer_sizes, libr_size, training_data, validation_data,
                 coefficient_threshold):
        self.layer_sizes = layer_sizes
        self.library_size = libr_size
        self.params, self.coefficient_mask = self.build_sindy_autoencoder(layer_sizes, libr_size)
        self.coefficient_threshold = coefficient_threshold

        #training data
        self.training_data = training_data
        self.validation_data = validation_data

        # monitor training
        self.reconstruction_loss = []
        self.sindy_z = []
        self.sindy_x = []

    @staticmethod
    def build_sindy_autoencoder(layer_sizes, library_size):

        encoding_params = build_encoder(layer_sizes)
        decoding_params = build_decoder(layer_sizes)

        sindy_coefficients = np.random.randn(library_size, layer_sizes[-1])
        coefficient_mask = np.ones((library_size, layer_sizes[-1]))
        params = {'encoder': encoding_params,
                  'decoder': decoding_params,
                  'sindy_coefficients': sindy_coefficients}
        return params, coefficient_mask

    @staticmethod
    def autoencoder_pass(params, coefficient_mask, x, dx):

        z = encoding_pass(params['encoder'], x)
        dz = z_derivative(params['encoder'], x, dx)
        x_decode = decoding_pass(params['decoder'], z)

        Theta = sindy_library(z, 3, 3)
        sindy_predict = np.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
        dx_decode = z_derivative_decode(params['decoder'], z, sindy_predict)

        return [x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict]

    def loss(self, sindy_autoencoder, iter):
        idx = batch_indices(iter)
        x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict = \
            self.autoencoder_pass(sindy_autoencoder,
                                  self.coefficient_mask,
                                  self.training_data['x'][idx, :],
                                  self.training_data['dx'][idx, :])

        reconstruction_loss = np.mean((x - x_decode) ** 2)
        sindy_z = np.mean((dz - sindy_predict) ** 2)
        sindy_x = np.mean((dx - dx_decode) ** 2)

        self.params = sindy_autoencoder
        self.reconstruction_loss.append(reconstruction_loss)
        self.sindy_z.append(sindy_z)
        self.sindy_x.append(sindy_x)

        total_loss = reconstruction_loss + 0 * sindy_z + 1e-4 * sindy_x
        return total_loss

    def reset_threshold(self):
        self.coefficient_mask = np.abs(self.params['sindy_coefficients']) > self.coefficient_threshold


if __name__ == "__main__":
    layer_sizes = [128, 64, 32, 3]
    batch_size = 8000
    epochs = 50

    lib_size = library_size(3, 3)

    noise_strength = 1e-6
    training_data = get_lorenz_data(1024, noise_strength=noise_strength)
    validation_data = get_lorenz_data(20, noise_strength=noise_strength)

    sindyautoencoder = SindyAutoencoder(layer_sizes, lib_size,
                                        training_data, validation_data, 0.1)


    num_batches = int(np.ceil(len(training_data['x']) / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    grad_fun = grad(sindyautoencoder.loss)
    print("     Epoch     |    Total Loss   |   Recon Loss  | Loss Sindy_z  ")
    def print_perf(params, iter , g):
        if iter % num_batches == 0:

            print("{:15}|{:20}".format(iter//num_batches, sindyautoencoder.loss(params, iter)))


    optimized_params = adam(grad_fun, sindyautoencoder.params, num_iters=epochs*num_batches, callback=print_perf,
                            step_size=1e-3)

    print("Reset coefficients")
