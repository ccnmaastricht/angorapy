from LSTM.fixedpointfinder import Adamfixedpointfinder
from LSTM.three_bit_flip_flop import Flipflopper
from LSTM.plot_utils import plot_fixed_points
import numpy as np

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
flopper.train(stim)
# visualize a single batch after training
flopper.visualize_flipflop(stim)

############################################################
# Initialize fpf and find fixed points
############################################################
# get weights and activations of trained model
weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
activations = flopper.get_activations(stim)

# initialize adam fpf
fpf = Adamfixedpointfinder(weights, flopper.hps['rnn_type'])
# sample states, i.e. a number of ICs
states = fpf.sample_states(activations, 4)
# generate corresponding input as zeros for flip flop task
# please that the input does not need to be zero for all tasks
inputs = np.zeros((states.shape[0], 3))
# find fixed points
fps = fpf.find_fixed_points(states, inputs)
# plot fixed points and state trajectories in 3D
plot_fixed_points(activations, fps, 4000)