import jax.numpy as jnp
from jax import random
from scipy.special import binom
from jax import vmap, value_and_grad, jit
from analysis.sindy_autoencoder.utils import sindy_library_jax
import matplotlib.pyplot as plt


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
    key = (k for k in key)

    encoding_params = build_encoder(layer_sizes, next(key))
    decoding_params = build_decoder(layer_sizes, next(key))

    control_encoding_params = build_encoder(layer_sizes, next(key))
    control_decoding_params = build_decoder(layer_sizes, next(key))

    sindy_coefficients = jnp.ones((library_size, layer_sizes[-1]))
    coefficient_mask = jnp.ones((library_size, layer_sizes[-1]))

    params = {'encoder': encoding_params,
              'decoder': decoding_params,
              'control_encoder': control_encoding_params,
              'control_decoder': control_decoding_params,
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


def control_autoencoder_pass(params, coefficient_mask, x, dx, u):

    z = encoding_pass(params['encoder'], x)
    x_decode = decoding_pass(params['decoder'], z)
    dz = z_derivative(params['encoder'], x, dx)

    y = encoding_pass(params['control_encoder'], u)
    u_decode = decoding_pass(params['control_decoder'], y)

    c = jnp.concatenate((z, y))
    Theta = sindy_library_jax(c, 2 * len(params['encoder'][-1][0]), 2)
    sindy_predict = jnp.matmul(Theta, coefficient_mask * params['sindy_coefficients'])
    dx_decode = z_derivative_decode(params['decoder'], z, sindy_predict)

    return [x_decode, u_decode, dz, sindy_predict, dx_decode]


def compute_latent_space(params, coefficient_mask, x, dx, u):
    z = encoding_pass(params['encoder'], x)
    dz = z_derivative(params['encoder'], x, dx)

    y = encoding_pass(params['control_encoder'], u)

    c = jnp.concatenate((z, y))
    Theta = sindy_library_jax(c, 2 * len(params['encoder'][-1][0]), 2)
    sindy_predict = jnp.matmul(Theta, coefficient_mask * params['sindy_coefficients'])

    return [z, y, sindy_predict]


batch_compute_latent_space = vmap(compute_latent_space, in_axes=({'encoder': None,
                                                                  'decoder': None,
                                                                  'control_encoder': None,
                                                                  'control_decoder': None,
                                                                  'sindy_coefficients': None},
                                                                 None, 0, 0, 0))
batch_control_autoencoder = vmap(control_autoencoder_pass, in_axes=({'encoder': None,
                                                                     'decoder': None,
                                                                     'control_encoder': None,
                                                                     'control_decoder': None,
                                                                     'sindy_coefficients': None},
                                                                     None, 0, 0, 0))


def loss(params, x, dx, u, coefficient_mask, hps):

    x_decode, u_decode, dz, sindy_predict, dx_decode = batch_control_autoencoder(params, coefficient_mask, x, dx, u)

    system_recon_loss = jnp.mean((x - x_decode)**2)
    control_recon_loss = jnp.mean((u - u_decode)**2)
    sindy_z_loss = jnp.mean((dz - sindy_predict)**2)
    sindy_x_loss = jnp.mean((dx - dx_decode)**2)
    sindy_regularization_loss = jnp.mean(jnp.abs(params['sindy_coefficients']))

    system_recon_loss = hps['system_loss_coeff'] * system_recon_loss
    control_recon_loss = hps['control_loss_coeff'] * control_recon_loss
    sindy_z_loss = hps['dz_loss_weight'] * sindy_z_loss
    sindy_x_loss = hps['dx_loss_weight'] * sindy_x_loss
    sindy_regularization_loss = hps['reg_loss_weight'] * sindy_regularization_loss
    total_loss = system_recon_loss + control_recon_loss + sindy_z_loss + sindy_x_loss + sindy_regularization_loss

    return {'total': total_loss,
            'sys_loss': system_recon_loss,
            'control_loss': control_recon_loss,
            'sindy_z_loss': sindy_z_loss,
            'sindy_x_loss': sindy_x_loss,
            'sindy_regularization_loss': sindy_regularization_loss}


def training_loss(params, x, dx, u, coefficient_mask, hps):
    return loss(params, x, dx, u, coefficient_mask, hps)['total']


def update(i, opt_state, opt_update, get_params, x, dx, u, coefficient_mask, hps):

    params = get_params(opt_state)
    value, grads = value_and_grad(training_loss)(params, x, dx, u, coefficient_mask, hps)

    return opt_update(i, grads, opt_state)


loss_jit = jit(loss)
update_jit = jit(update, static_argnums=(2, 3))


def plot_params(params, coefficient_mask):

    plt.figure()
    plt.subplot(121)
    plt.imshow(params['sindy_coefficients'])
    plt.axis('off')
    plt.title('SINDy Coefficients')

    plt.subplot(122)
    plt.imshow(coefficient_mask)
    plt.axis('off')
    plt.title('Coefficient Mask')


'''
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
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)'''

'''                        
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
    self.refinement_loss = state['history']['refinement_loss']'''