from fixedpointfinder.three_bit_flip_flop import Flipflopper
import numpy as np
from fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from rnnconstruction.rnnconcstruct import Rnnconstructor
from rnnconstruction.plot_utils import plot_history
from fixedpointfinder.plot_utils import plot_fixed_points

############################################################
# Create and train recurrent model on 3-Bit FlipFop task
############################################################
# specify architecture e.g. 'vanilla' and number of hidden units
rnn_type = 'gru'
n_hidden = 24
# initialize Flipflopper class
flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
# generate trials
stim = flopper.generate_flipflop_trials()
# train the model
# first_history = flopper.train(stim, 4000, save_model=True)
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
                           alr_decayr=0.001,
                           max_iters=10000)
# sample states, i.e. a number of ICs
states = fpf.sample_states(activations, 1024)
# vel = fpf.compute_velocities(np.hstack(activations[1:]), np.zeros((32768, 3)))
# generate corresponding input as zeros for flip flop task
# please that the input does not need to be zero for all tasks
inputs = np.zeros((states.shape[0], 3))
# find fixed points
fps = fpf.find_fixed_points(states, inputs)
# get fps to have points to train for
plot_fixed_points(activations, fps, 2000, 1)


reco = Rnnconstructor(fps, n_hidden, rnn_type,
                      epsilon=0.01,
                      alr_decayr=0.0001,
                      max_iters=10000)

recurrentweights = reco.train_recurrentweights(flopper.weights[1])
weights[1] = recurrentweights
fph = reco.compute_jacobians(fps, weights, inputs)
plot_fixed_points(activations, fph, 2000, 4)


# retrained_history, retrained_model = flopper.train_pretrained(stim, 2000, weights, recurrentweights, False)
# score = flopper.pretrained_predict(retrained_model, stim)

# plot_history(retrained_history)