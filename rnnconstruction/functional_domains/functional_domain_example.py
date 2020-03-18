from fixedpointfinder.three_bit_flip_flop import Flipflopper
import autograd.numpy as np
import os
from rnnconstruction.functional_domains.plot_utils import plot_functional_domains
import tensorflow as tf
from rnnconstruction.functional_domains.functional_domain_analysis import FDA
from fixedpointfinder.minimization import adam_weights_optimizer
import itertools as it

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




