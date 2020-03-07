from fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from fixedpointfinder.three_bit_flip_flop import Flipflopper
from fixedpointfinder.plot_utils import plot_fixed_points, plot_velocities, visualize_flipflop
import numpy as np
import os
import itertools as it
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import tensorflow as tf
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
activations = flopper.get_activations(stim)
# initialize adam fpf
# fpf = Adamfixedpointfinder(weights, rnn_type,
                       #    q_threshold=1e-12,
                       #    epsilon=0.01,
                       #    alr_decayr=0.0001,
                       #    max_iters=5000)
# sample states, i.e. a number of ICs
# states = fpf.sample_states(activations, 1000, 0.5)
# vel = fpf.compute_velocities(np.hstack(activations[1:]), np.zeros((32768, 3)))
# generate corresponding input as zeros for flip flop task
# please keep in mind that the input does not need to be zero for all tasks
# inputs = np.zeros((states.shape[0], 3))
# find fixed points
# fps = fpf.find_fixed_points(states, inputs)
# plot fixed points and state trajectories in 3D
# plot_fixed_points(activations, fps, 3000, 1)

# find neurons that wire together and or represent certain features in the input
import sklearn.decomposition as skld

output_first_batch = stim['output'][0, :, :]
activations_first_episode = activations[:256, :]

activations = np.vstack(activations)
pca = skld.PCA(3)
pca.fit(activations)
X_pca = pca.transform(activations)
loadings = pca.components_

loadings = loadings.transpose()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(loadings[:, 0], loadings[:, 1], loadings[:, 2])


output_weights = flopper.model.get_layer('dense').get_weights()

activations_reconstructed = np.matmul(output_first_batch, output_weights[0].transpose())

def classify_neurons(output_weights):
    weights_by_number = {'number_one': {'weights': [], 'index': []},
                         'number_two': {'weights': [], 'index': []},
                         'number_three': {'weights': [], 'index': []}}
    k = 0

    for weights_per_neuron in output_weights:
        big_weights = np.abs(weights_per_neuron) > 0.15
        if big_weights[0]:
            weights_by_number['number_one']['weights'].append(weights_per_neuron)
            weights_by_number['number_one']['index'].append(k)
        elif big_weights[1]:
            weights_by_number['number_two']['weights'].append(weights_per_neuron)
            weights_by_number['number_two']['index'].append(k)
        elif big_weights[2]:
            weights_by_number['number_three']['weights'].append(weights_per_neuron)
            weights_by_number['number_three']['index'].append(k)
        k +=1
    return weights_by_number

weights_by_number = classify_neurons(output_weights[0])
