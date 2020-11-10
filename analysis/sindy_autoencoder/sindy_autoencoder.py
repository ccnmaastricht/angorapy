from jax import vmap, random, jit, value_and_grad
import jax.numpy as jnp
import jax
from jax.experimental import optimizers
import time

from analysis.sindy_autoencoder.lorenz_example import get_lorenz_data
from analysis.sindy_autoencoder.utils import build_sindy_autoencoder, autoencoder_pass, library_size


class SindyAutoencoder(object):

    def __init__(self,
                 layer_sizes: list,
                 poly_order: int,
                 seed: int = 1,
                 sequential_thresholding: bool = True,
                 coefficient_threshold: float = 0.1,
                 threshold_frequency: int = 500,
                 recon_loss_weight: float = 1.0,
                 sindy_z_loss_weight: float = 0.0,
                 sindy_x_loss_weight: float = 1e-4,
                 sindy_regularization_loss_weight: float = 1e-5,
                 max_epochs: int = 5000,
                 batch_size: int = 8000,
                 learning_rate: float = 1e-3,
                 print_updates: bool = True):

        self.layer_sizes = layer_sizes
        self.poly_order = poly_order
        self.library_size = library_size(layer_sizes[-1], poly_order)

        self.key = random.PRNGKey(seed)
        self.autoencoder, self.coefficient_mask = build_sindy_autoencoder(layer_sizes, self.key)
        self.vmap_autoencoder = vmap(autoencoder_pass, in_axes=({'encoder': None,
                                                                 'decoder': None,
                                                                 'sindy_coefficients': None},
                                                                None, 0, 0))

        self.recon_loss_weight = recon_loss_weight
        self.sindy_z_loss_weight = sindy_z_loss_weight
        self.sindy_x_loss_weight = sindy_x_loss_weight
        self.sindy_regularization_loss_weight = sindy_regularization_loss_weight

        loss, refinement_loss = self.build_loss_fun()
        self.loss = jit(loss, device=jax.devices()[0])

        #optimizer
        self.learning_rate = learning_rate
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # thresholding for coefficient mask
        self.sequential_thresholding = sequential_thresholding
        self.thresholding_frequency = threshold_frequency
        self.coefficient_threshold = coefficient_threshold

        self.print_updates = print_updates

        self.train_loss = []

    def build_loss_fun(self):

        vmap_autoencoder = self.vmap_autoencoder

        recon_loss_weight = self.recon_loss_weight
        sindy_z_loss_weight = self.sindy_z_loss_weight
        sindy_x_loss_weight = self.sindy_x_loss_weight
        sindy_regularization_loss_weight = self.sindy_regularization_loss_weight

        def loss(sindy_autoencoder, coefficient_mask, input, dx_input):
            x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict = vmap_autoencoder(sindy_autoencoder, coefficient_mask, input,
                                                                             dx_input)

            reconstruction_loss = jnp.mean((x - x_decode) ** 2)
            sindy_z = jnp.mean((dz - sindy_predict) ** 2)
            sindy_x = jnp.mean((dx - dx_decode) ** 2)
            sindy_regularization = jnp.mean(jnp.abs(sindy_autoencoder['sindy_coefficients']))

            total_loss = recon_loss_weight * reconstruction_loss + \
                         sindy_z_loss_weight * sindy_z + \
                         sindy_x_loss_weight * sindy_x + \
                         sindy_regularization_loss_weight * sindy_regularization
            return total_loss

        def refinement_loss(sindy_autoencoder, coefficient_mask, input, dx_input):
            x, dx, z, dz, x_decode, dx_decode, Theta, sindy_predict = vmap_autoencoder(sindy_autoencoder, coefficient_mask, input,
                                                                             dx_input)

            reconstruction_loss = jnp.mean((x - x_decode) ** 2)
            sindy_z = jnp.mean((dz - sindy_predict) ** 2)
            sindy_x = jnp.mean((dx - dx_decode) ** 2)

            refinement_loss_total = recon_loss_weight * reconstruction_loss + \
                                    sindy_z_loss_weight * sindy_z + \
                                    sindy_x_loss_weight * sindy_x
            return refinement_loss_total

        return loss, refinement_loss

    def update(self, params, coefficient_mask, input, dx_input, opt_state):
        value, grads = value_and_grad(self.loss)(params, coefficient_mask, input, dx_input)
        opt_state = self.opt_update(0, grads, opt_state)
        return self.get_params(opt_state), opt_state, value

    def train(self, training_data: dict):

        opt_state = self.opt_init(self.autoencoder)
        params = self.get_params(opt_state)

        num_batches = int(jnp.ceil(len(training_data['x']) / self.batch_size))

        def batch_indices(iter):
            idx = iter % num_batches
            return slice(idx * batch_size, (idx + 1) * self.batch_size)

        for epoch in range(self.max_epochs):
            start = time.time()
            for i in range(num_batches):
                idx = batch_indices(i)

                params, opt_state, value = self.update(params, self.coefficient_mask,
                                                       training_data['x'][idx, :],
                                                       training_data['dx'][idx, :],
                                                       opt_state)
                self.train_loss.append(value)
            if epoch % self.thresholding_frequency == 0 and epoch > 1 and self.sequential_thresholding:
                self.coefficient_mask = jnp.abs(params['sindy_coefficients']) > 0.1
                print("Updated coefficient mask")

            stop = time.time()
            print("Epoch", str(epoch), "Loss:", str(value), "This took:", str(jnp.round(stop - start, 4)), "s")

    def validate(self):
        pass

    def simulate(self):
        pass


if __name__ == "__main__":
    layer_sizes = [128, 64, 32, 3]
    seed = 1
    SA = SindyAutoencoder(layer_sizes, poly_order=3, seed=seed, max_epochs=10)

    noise_strength = 1e-6
    training_data = get_lorenz_data(1024, noise_strength=noise_strength)
    # validation_data = get_lorenz_data(20, noise_strength=noise_strength)

    batch_size = 8000

    SA.train(training_data=training_data)


