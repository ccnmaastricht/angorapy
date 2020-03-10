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

def reconstruct_from_evecs(evals, evecs, threshold, bigger):
    if bigger:
        big_evals = np.abs(evals) > threshold
    else:
        big_evals = np.abs(evals) < threshold
    new_evecs = np.zeros((24, 24))
    new_evecs[:, big_evals] = evecs[:, big_evals]

    diagonal_evals = np.zeros((24, 24))
    diag_indices = np.diag_indices(24)
    evals_vector = np.zeros(24)
    evals_vector[big_evals] = evals[big_evals]
    diagonal_evals[diag_indices] = evals_vector

    reconstructed_weights = np.dot(new_evecs,  diagonal_evals * new_evecs.T)

    return reconstructed_weights


bigger = True
reconstructed_weights = reconstruct_from_evecs(evals, evecs, 0.1, bigger)
reconstructed_weights_small = reconstruct_from_evecs(evals, evecs, 0.1, True)
weights[1] = reconstructed_weights