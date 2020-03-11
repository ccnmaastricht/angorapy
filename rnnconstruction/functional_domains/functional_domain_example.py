from fixedpointfinder.three_bit_flip_flop import Flipflopper
import autograd.numpy as np
import os
from rnnconstruction.functional_domains.plot_utils import plot_functional_domains
import tensorflow as tf
from rnnconstruction.functional_domains.functional_domain_analysis import FDA
from fixedpointfinder.minimization import adam_weights_optimizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############################################################
# Create and train recurrent model on 3-Bit FlipFop task
############################################################
# specify architecture e.g. 'vanilla' and number of hidden units
rnn_type = 'vanilla'
n_hidden = 24

# initialize Flipflopper class
flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
# generate trials
stim = flopper.generate_flipflop_trials()
# train the model
# flopper.train(stim, 2000, save_model=True)

# visualize a single batch after training
# prediction = flopper.model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))
# visualize_flipflop(stim)

# if a trained model has been saved, it may also be loaded
flopper.load_model()
############################################################
# Initialize fpf and find fixed points
############################################################
# get weights and activations of trained model
weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
output_weights = flopper.model.get_layer('dense').get_weights()
activations = flopper.get_activations(stim)

# find neurons that wire together and or represent certain features in the input
activations = np.vstack(activations)

fda = FDA(output_weights[0], 1, n_hidden)

initial_activations = np.zeros((1, 24))
fake_activations = np.vstack((initial_activations, activations[:-1, :]))
inputs = np.vstack(stim['inputs'])

recurrent_layer, dense_layer, pooling_layer = fda.reconstruct_model_with_domains(weights)

recurrent_activations = recurrent_layer(inputs, fake_activations)
generated_outputs = dense_layer(recurrent_activations)

recurrent_activations_stacked = np.hstack(recurrent_activations)
pooling_weights = np.random.randn(9, 3) * 1e-03
recurrent_output_weights = np.random.randn(recurrent_activations_stacked.shape[1], n_hidden) * 1e-03
outputs = np.vstack(stim['output'])


def recurrent_pooling_layer(recurrent_output_weights):
    return np.matmul(recurrent_activations_stacked, recurrent_output_weights)

def objective_function(x):

    return np.mean(np.square(- np.matmul(recurrent_pooling_layer(x), output_weights[0]) + outputs))

bit = 'first_bit'
activations_per_bit = fda.reconstruct_domains(activations, outputs, bit)
plot_functional_domains(activations, activations_per_bit)

pooling_weights = adam_weights_optimizer(objective_function, recurrent_output_weights, 0,
                                         epsilon=0.01,
                                         alr_decayr=0.001,
                                         max_iter=5000,
                                         print_every=200,
                                         init_agnc=1.0,
                                         agnc_decayr=0.0001,
                                         verbose=True)

orthonormalbasis = np.linalg.qr(weights[1])
evals, evecs = np.linalg.eig(weights[1])
diagonal_evals = np.diag(evals)

A = evals[4].real
B = evals[4].imag

cosval = A/np.sqrt(np.square(A) + np.square(B))
sinval = B/np.sqrt(np.square(A) + np.square(B))
r = np.sqrt(np.square(A) + np.square(B))

import math
def polar(z):
    a= z.real
    b= z.imag
    r = math.hypot(a,b)
    theta = math.atan2(b,a)
    return r,theta # use return instead of print.


# theta almost 120°!!!
degrees = []
real_evals = []
for val in evals:
    if type(val) == np.complex64:
        r, theta = polar(val)
        degrees.append(np.degrees(theta))
        print(np.degrees(theta))
        if theta == 0:
            real_evals.append(val)

degrees = np.vstack(degrees)
degrees = degrees[degrees != 0]

combinations = []
for a, b in it.combinations(degrees, 2):
        combinations.append(a + b)

combinations = np.vstack(combinations)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

degrees_fixedpoints = []
import itertools as it
for a, b in it.combinations(fps, 2):
        pointa = a['x']
        pointb = b['x']
        print(np.degrees(angle_between(pointa, pointb)))
        degrees_fixedpoints.append(np.degrees(angle_between(pointa, pointb)))
        print(np.linalg.norm(pointa))
degrees_fixedpoints = np.vstack(degrees_fixedpoints)

list_combinations_and_degrees = []
for combination in combinations:
    for degrees_fixedpoint in degrees_fixedpoints:
        if np.abs(combination-degrees_fixedpoint) < 0.5:
            list_combinations_and_degrees.append((degrees_fixedpoint, combination))



bigger = True
reconstructed_weights = reconstruct_from_evecs(evals, evecs, 0.1, bigger)
reconstructed_weights_small = reconstruct_from_evecs(evals, evecs, 0.1, True)
# weights[1] = reconstructed_weights

# serialize the recurrent layer to proof that eigenvectors do what they do
input_to_recurrent_layer = np.matmul(np.vstack(stim['inputs']), weights[0])
fake_h = np.vstack((np.zeros(24), activations[:-1, :]))
def recurrent_layer_serialized(reconstructed_matrices, input_recurrent_layer, weights, fake_h):
    h = fake_h
    # h = weights[2] + input_recurrent_layer
    for i in range(1):
        h = h @ reconstructed_matrices[i] #weights[1]

    h = np.tanh(h + input_recurrent_layer + weights[2])
    return h


def after_serialized_layer(h, output_weights):

    output_network = np.matmul(h, output_weights[0]) + output_weights[1]

    return output_network

h = recurrent_layer_serialized(reconstructed_matrices, input_to_recurrent_layer, weights, fake_h)

network_output = after_serialized_layer(h, output_weights)

mse = np.mean(np.sqrt(np.square(network_output-outputs)))
print(mse)