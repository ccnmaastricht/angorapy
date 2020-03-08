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
X_pca = pca.transform(activations_first_number)
loadings = pca.components_

loadings = loadings.transpose()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
plt.ylim((-2, 2)), plt.xlim((-2, 2))
ax.set_zlim(-2, 2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(loadings[:, 0], loadings[:, 1], loadings[:, 2])


output_weights = flopper.model.get_layer('dense').get_weights()

activations_reconstructed = np.matmul(output_first_batch, output_weights[0].transpose())

def classify_neurons(output_weights, threshold):
    weights_by_number = {'number_one': {'weights': [], 'index': []},
                         'number_two': {'weights': [], 'index': []},
                         'number_three': {'weights': [], 'index': []}}
    k = 0

    for weights_per_neuron in output_weights:
        big_weights = np.abs(weights_per_neuron)
        biggest_weight = np.max(big_weights)
        difference = biggest_weight - big_weights
        ratio_biggest_to_weights = difference/biggest_weight
        big_weights = ratio_biggest_to_weights < threshold
        for i in range(len(big_weights)):
            if big_weights[i] and i == 0:
                weights_by_number['number_one']['weights'].append(weights_per_neuron)
                weights_by_number['number_one']['index'].append(k)
            elif big_weights[i] and i == 1:
                weights_by_number['number_two']['weights'].append(weights_per_neuron)
                weights_by_number['number_two']['index'].append(k)
            elif big_weights[i] and i == 2:
                weights_by_number['number_three']['weights'].append(weights_per_neuron)
                weights_by_number['number_three']['index'].append(k)
        k +=1
    return weights_by_number

def unique_neurons(weights_by_number):
    pass

def reconstruct_model_with_domains(weights, weights_by_number):

    def split_recurrent_kernel(weights_by_number):
        first_bit_index = np.asarray(weights_by_number['number_one']['index'])
        second_bit_index = np.asarray(weights_by_number['number_two']['index'])
        third_bit_index = np.asarray(weights_by_number['number_two']['index'])

        return [first_bit_index, second_bit_index, third_bit_index]

    def recurrent_layer(inputs, activations):
        inputweights, recurrentweights, recurrentbias = weights[0], weights[1], weights[2]
        list_of_indices = split_recurrent_kernel(weights_by_number)

        first_bit_inputweights, first_bit_recurrentweights, first_bit_recurrentbias = inputweights[:, list_of_indices[0]], \
                                                                                      recurrentweights[list_of_indices[0], :], \
                                                                                      recurrentbias[list_of_indices[0]]
        first_bit_recurrentweights = first_bit_recurrentweights[:, list_of_indices[0]]

        second_bit_inputweights, second_bit_recurrentweights, second_bit_recurrentbias = inputweights[:, list_of_indices[1]], \
                                                                                      recurrentweights[list_of_indices[1], :], \
                                                                                      recurrentbias[list_of_indices[1]]
        second_bit_recurrentweights = second_bit_recurrentweights[:, list_of_indices[1]]

        third_bit_inputweights, third_bit_recurrentweights, third_bit_recurrentbias = inputweights[:, list_of_indices[2]], \
                                                                                      recurrentweights[list_of_indices[2], :], \
                                                                                      recurrentbias[list_of_indices[2]]
        third_bit_recurrentweights = third_bit_recurrentweights[:, list_of_indices[2]]

        first_bit_activations = activations[:, list_of_indices[0]]
        second_bit_activations = activations[:, list_of_indices[1]]
        third_bit_activations = activations[:, list_of_indices[2]]
        first_bit_output = np.tanh(np.matmul(first_bit_activations, first_bit_recurrentweights) + \
                                   np.matmul(inputs, first_bit_inputweights) + first_bit_recurrentbias)

        second_bit_output = np.tanh(np.matmul(second_bit_activations, second_bit_recurrentweights) + \
                                    np.matmul(inputs, second_bit_inputweights) + second_bit_recurrentbias)

        third_bit_output = np.tanh(np.matmul(third_bit_activations, third_bit_recurrentweights) + \
                                   np.matmul(inputs, third_bit_inputweights) + third_bit_recurrentbias)
        return [first_bit_output, second_bit_output, third_bit_output]

    def dense_layer(recurrent_activations):
        first_bit_weights = np.vstack(weights_by_number['number_one']['weights'])
        second_bit_weights = np.vstack(weights_by_number['number_two']['weights'])
        third_bit_weights = np.vstack(weights_by_number['number_two']['weights'])

        first_bit_output = np.matmul(recurrent_activations[0], first_bit_weights)
        second_bit_output = np.matmul(recurrent_activations[1], second_bit_weights)
        third_bit_output = np.matmul(recurrent_activations[2], third_bit_weights)

        return first_bit_output + second_bit_output + third_bit_output

    return recurrent_layer, dense_layer

weights_by_number = classify_neurons(output_weights[0], 0.5)

initial_activations = np.zeros((1, 24))
fake_activations = np.vstack((initial_activations, activations[:-1, :]))

inputs = np.vstack(stim['inputs'])

recurrent_layer, dense_layer = reconstruct_model_with_domains(weights, weights_by_number)

recurrent_activations = recurrent_layer(inputs, fake_activations)
generated_outputs = dense_layer(recurrent_activations)

mse = (np.square(outputs - generated_outputs)).mean(axis=1)

avemse = np.mean(mse)
print(avemse)



# mean_neuron_activations = np.mean(activations, axis=0)
activations_first_number = np.zeros(activations.shape)
# activations_first_number = np.repeat(np.reshape(mean_neuron_activations, (-1, 24)), activations.shape[0], axis=0)
first_bit_neurons = np.asarray(weights_by_number['number_three']['index'])
activations_first_number[:, first_bit_neurons] = activations[:, first_bit_neurons]

first_bit_neuron_weights = np.vstack(weights_by_number['number_three']['weights'])

outputs = np.vstack(stim['output'])
firstbit_neuron_activations_reconstructed = np.matmul(outputs, first_bit_neuron_weights.transpose())

activations_first_number[:, first_bit_neurons] = firstbit_neuron_activations_reconstructed
