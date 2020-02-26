from fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from fixedpointfinder.three_bit_flip_flop import Flipflopper
from fixedpointfinder.plot_utils import plot_fixed_points, plot_velocities
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############################################################
# Create and train recurrent model on 3-Bit FlipFop task
############################################################
# specify architecture e.g. 'vanilla' and number of hidden units
rnn_type = 'lstm'
n_hidden = 24

# initialize Flipflopper class
flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
# generate trials
stim = flopper.generate_flipflop_trials()
# train the model
# flopper.train(stim, 2000, save_model=True)
# visualize a single batch after training
# flopper.visualize_flipflop(stim)
# if a trained model has been saved, it may also be loaded
flopper.load_model()
############################################################
# Initialize fpf and find fixed points
############################################################
# get weights and activations of trained model
weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
activations = flopper.get_activations(stim)
# initialize adam fpf
fpf = Adamfixedpointfinder(weights, rnn_type,
                           q_threshold=1e-12,
                           epsilon=0.001,
                           alr_decayr=0.0001,
                           max_iters=5000)
# sample states, i.e. a number of ICs
states = fpf.sample_states(activations, 500, 0.5)
# vel = fpf.compute_velocities(np.hstack(activations[1:]), np.zeros((32768, 3)))
# generate corresponding input as zeros for flip flop task
# please keep in mind that the input does not need to be zero for all tasks
inputs = np.zeros((states.shape[0], 3))
# find fixed points
fps = fpf.find_fixed_points(states, inputs)
# plot fixed points and state trajectories in 3D
if rnn_type == 'lstm':
    activations = np.hstack(activations[1:])
plot_fixed_points(activations, fps, 3000, 4)

