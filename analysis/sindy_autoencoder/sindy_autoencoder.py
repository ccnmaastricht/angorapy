from jax import vmap, random, jit, value_and_grad, jacobian
import jax.numpy as jnp
import jax
from jax.experimental import optimizers
import time
import gym
import pickle
import os

from analysis.sindy_autoencoder.lorenz_example import get_lorenz_data
from analysis.sindy_autoencoder.utils import build_sindy_autoencoder, autoencoder_pass, \
    library_size, simulation_pass, encoding_pass, sindy_library_jax


class SindyAutoencoder(object):
    """Class for implementation of SindyAutoencoder or
    Data-driven discovery of coordinates and governing equations

    paper: https://doi.org/10.1073/pnas.1906995116
    code: https://github.com/kpchamp/SindyAutoencoders

    This is an implementation in JAX instead of Tensorflow in the original publication.

    Args:
        layer_sizes (list): list of int specifying the layer sizes for the autoencoder. Last layer should be latend_dim,i.e. size of z
        poly_order (int): polynomial order in Theta
        seed (int): seed for PRNGkey
        sequential_thresholding (bool): whether to do sequential thresholding using coefficient_mask
        coefficient_threshold (float): theshold below which in absolute value sindy_coefficients will be 0
        threshold_frequency (int): number of epochs to apply thresholding after
        recon_loss_weight (float): weight multiplier for reconstruction loss
        sindy_z_loss_weight (float): weight multiplier for sindy_z loss
        sindy_x_loss_weight (float): weight multiplier for sindy_x loss
        sindy_regularization_loss_weight (float): weight multiplier for sindy regularization loss
        max_epochs (int): maximum number of epochs to be run
        refinement_epochs (int): number of refinement epochs (no regularization loss)
        batch_size (int): batch_size for one iteration in one epoch
        learning_rate (float): step size parameter for updating parameters
        print_updates (bool): switch on/off printing of updates during training


    """
    def __init__(self,
                 layer_sizes: list,
                 poly_order: int,
                 seed: int = 1,
                 sequential_thresholding: bool = True,
                 coefficient_threshold: float = 0.1,
                 threshold_frequency: int = 500,
                 use_sine: bool = False,
                 recon_loss_weight: float = 1.0,
                 sindy_z_loss_weight: float = 0.0,
                 sindy_x_loss_weight: float = 1e-4,
                 sindy_regularization_loss_weight: float = 1e-5,
                 max_epochs: int = 5000,
                 refinement_epochs: int = 1000,
                 batch_size: int = 8000,
                 learning_rate: float = 1e-3,
                 print_updates: bool = True):

        self.layer_sizes = layer_sizes
        self.poly_order = poly_order
        self.library_size = library_size(layer_sizes[-1], poly_order, use_sine=use_sine)

        self.key = random.PRNGKey(seed)
        self.autoencoder, self.coefficient_mask = build_sindy_autoencoder(layer_sizes,
                                                                          self.library_size,
                                                                          self.key)
        # TODO: add selector for activation functions
        self.vmap_autoencoder_pass = vmap(autoencoder_pass, in_axes=({'encoder': None,
                                                                      'decoder': None,
                                                                      'sindy_coefficients': None},
                                                                     None, 0, 0))
        self.simulation_pass = simulation_pass

        self.recon_loss_weight = recon_loss_weight
        self.sindy_z_loss_weight = sindy_z_loss_weight
        self.sindy_x_loss_weight = sindy_x_loss_weight
        self.sindy_regularization_loss_weight = sindy_regularization_loss_weight

        self.loss_fun = jit(self.loss, device=jax.devices()[0])

        #optimizer
        self.learning_rate = learning_rate
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.max_epochs = max_epochs
        self.refinement_epochs = refinement_epochs
        self.batch_size = batch_size

        # thresholding for coefficient mask
        self.sequential_thresholding = sequential_thresholding
        self.thresholding_frequency = threshold_frequency
        self.coefficient_threshold = coefficient_threshold

        self.print_updates = print_updates

        self.train_loss = []
        self.refinement_loss = []

    def loss(self, sindy_autoencoder, coefficient_mask, x, dx):
        x, dx, dz, x_decode, dx_decode, sindy_predict = self.vmap_autoencoder_pass(sindy_autoencoder,
                                                                                   coefficient_mask,
                                                                                   x,
                                                                                   dx)

        reconstruction_loss = jnp.mean((x - x_decode) ** 2)
        sindy_z = jnp.mean((dz - sindy_predict) ** 2)
        sindy_x = jnp.mean((dx - dx_decode) ** 2)
        sindy_regularization = jnp.mean(jnp.abs(sindy_autoencoder['sindy_coefficients']))

        total_loss = self.recon_loss_weight * reconstruction_loss + \
                     self.sindy_z_loss_weight * sindy_z + \
                     self.sindy_x_loss_weight * sindy_x + \
                     self.sindy_regularization_loss_weight * sindy_regularization
        return total_loss

    def update(self, params, coefficient_mask, X, dx, opt_state):
        value, grads = value_and_grad(self.loss_fun)(params, coefficient_mask, X, dx)
        opt_state = self.opt_update(0, grads, opt_state)
        return self.get_params(opt_state), opt_state, value

    def train(self, training_data: dict):
        n_updates = 0

        opt_state = self.opt_init(self.autoencoder)
        params = self.get_params(opt_state)

        num_batches = int(jnp.ceil(len(training_data['x']) / self.batch_size))
        batch_size = self.batch_size

        def batch_indices(iter):
            idx = iter % num_batches
            return slice(idx * batch_size, (idx + 1) * batch_size)
        print("TRAINING...")
        for epoch in range(self.max_epochs):
            start = time.time()
            for i in range(num_batches):
                id = batch_indices(i)

                params, opt_state, value = self.update(params, self.coefficient_mask,
                                                       training_data['x'][id, :],
                                                       training_data['dx'][id, :],
                                                       opt_state)
                self.train_loss.append(value)
                n_updates += 1

            if epoch % self.thresholding_frequency == 0 and epoch > 1 and self.sequential_thresholding:
                self.coefficient_mask = jnp.abs(params['sindy_coefficients']) > 0.1
                print("Updated coefficient mask")

            stop = time.time()
            dt = stop - start
            if self.print_updates:
                self.print_update(epoch, n_updates, dt)

        print("REFINEMENT...")
        self.sindy_regularization_loss_weight = 0.0  # no regularization anymore

        for ref_epoch in range(self.refinement_epochs):
            start = time.time()
            for i in range(num_batches):
                id = batch_indices(i)

                params, opt_state, value = self.update(params, self.coefficient_mask,
                                                       training_data['x'][id, :],
                                                       training_data['dx'][id, :],
                                                       opt_state)
                self.refinement_loss.append(value)
                n_updates += 1

            stop = time.time()
            dt = stop - start
            if self.print_updates:
                self.print_update(ref_epoch, n_updates, dt)

        print(f"FINISHING...\n"
              f"Sparsity: {jnp.sum(self.coefficient_mask)} active terms")

        self.autoencoder = params

    def validate(self):
        pass

    def simulate_dynamics(self, x, dx):
        x, dx, z_next, sindy_predict, x_next, dx_next = self.simulation_pass(self.autoencoder,
                                                                             self.coefficient_mask,
                                                                             x,
                                                                             dx)

        return z_next, sindy_predict, x_next

    def simulate_episode(self, env: gym.Env):
        pass

    def evaluate_jacobian(self, x):
        z = encoding_pass(self.autoencoder['encoder'], x)
        print(z)

        def dynamics_fun(x):
            return jnp.matmul(sindy_library_jax(x, self.layer_sizes[-1], self.poly_order), \
                        self.coefficient_mask * self.autoencoder['sindy_coefficients'])

        jacobian_fun = jacobian(dynamics_fun)

        jacobian_evaluation = jacobian_fun(z)

        return jacobian_evaluation

    def save_state(self, filename, save_dir: str = ''):
        state = {'autoencoder': self.autoencoder,
                 'coefficient_mask': self.coefficient_mask,
                 'hps': {'layers': self.layer_sizes,
                         'poly_order': self.poly_order,
                         'library:size': self.library_size,
                         'lr': self.learning_rate,
                         'epochs': self.max_epochs,
                         'batch_size': self.batch_size,
                         'sequential_threshold': self.sequential_thresholding,
                         'thresholding_frequency': self.thresholding_frequency,
                         'threshold_coefficient': self.coefficient_threshold},
                 'history': {'train_loss': self.train_loss,
                             'refinement_loss': self.refinement_loss}}

        try:
            directory = os.getcwd() + '/analysis/sindy_autoencoder/' + save_dir + filename + '.pkl'

            with open(file=directory, mode='wb') as f:
                pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

        except FileNotFoundError:
            directory = '/analysis/sindy_autoencoder/' + save_dir + filename + '.pkl'

            with open(file=directory, mode='wb') as f:
                pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    def load_state(self, filename, save_dir: str = ''):
        try:
            directory = os.getcwd() + '/analysis/sindy_autoencoder/' + save_dir + filename + '.pkl'
            with open(directory, 'rb') as f:
                state = pickle.load(f)

        except FileNotFoundError:
            directory = '/analysis/sindy_autoencoder/' + save_dir + filename + '.pkl'
            with open(directory, 'rb') as f:
                state = pickle.load(f)

        self.autoencoder = state['autoencoder']
        self.coefficient_mask = state['coefficient_mask']
        self.layer_sizes = state['hps']['layers']
        self.poly_order = state['hps']['poly_order']
        self.library_size = state['hps']['library:size']
        self.learning_rate = state['hps']['lr']
        self.max_epochs = state['hps']['epochs']
        self.batch_size = state['hps']['batch_size']
        self.sequential_thresholding = state['hps']['sequential_threshold']
        self.thresholding_frequency = state['hps']['thresholding_frequency']
        self.coefficient_threshold = state['hps']['threshold_coefficient']
        self.train_loss = state['history']['train_loss']
        self.refinement_loss = state['history']['refinement_loss']

    def print_update(self, epoch, n_updates, dt):
        print(f"Epoch {1 + epoch}",
              f"| Loss {round(self.train_loss[-1], 7)}",
              f"| Updates {n_updates}",
              f"| This took: {round(dt, 4)}s")


# TODO: add model orders higher than 1
# TODO: add different initializations for weights and sindy coefficients

class SindyControlAutoencoder(SindyAutoencoder):

    def __init__(self):
        pass


if __name__ == "__main__":
    layers = [128, 64, 32, 3]
    seed = 1
    SA = SindyAutoencoder(layers, poly_order=2, seed=seed, max_epochs=3000,
                          refinement_epochs=200)

    noise_strength = 1e-6
    training_data = get_lorenz_data(1024, noise_strength=noise_strength)
    # validation_data = get_lorenz_data(20, noise_strength=noise_strength)

    SA.train(training_data=training_data)
    SA.save_state(filename='LorenzSystem')



