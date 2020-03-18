import autograd.numpy as np

class FDA:

    def __init__(self, output_weights, classification_threshold, n_hidden):
        self.weights_by_number = self._classify_neurons(output_weights, classification_threshold)
        self.n_hidden = n_hidden

    def _classify_neurons(self, output_weights, threshold):
        weights_by_number = {'first_bit': {'weights': [], 'index': []},
                             'second_bit': {'weights': [], 'index': []},
                             'third_bit': {'weights': [], 'index': []}}
        k = 0

        for weights_per_neuron in output_weights:
            big_weights = np.abs(weights_per_neuron)
            # big_weights = big_weights > threshold
            biggest_weight = np.max(big_weights)
            difference = biggest_weight - big_weights
            ratio_biggest_to_weights = difference / biggest_weight
            big_weights = ratio_biggest_to_weights < threshold
            for i in range(len(big_weights)):
                if big_weights[i] and i == 0:
                    weights_by_number['first_bit']['weights'].append(weights_per_neuron)
                    weights_by_number['first_bit']['index'].append(k)
                elif big_weights[i] and i == 1:
                    weights_by_number['second_bit']['weights'].append(weights_per_neuron)
                    weights_by_number['second_bit']['index'].append(k)
                elif big_weights[i] and i == 2:
                    weights_by_number['third_bit']['weights'].append(weights_per_neuron)
                    weights_by_number['third_bit']['index'].append(k)
            k += 1
        return weights_by_number

    def reconstruct_model_with_domains(self, weights):

        def recurrent_layer(inputs, activations):
            split_inputweights, split_recurrentweights, split_recurrentbias = \
                self.get_recurrent_subkernel(weights)
            split_activations = self.split_recurrent_activations(activations)

            first_bit_output = np.tanh(np.matmul(split_activations[0], split_recurrentweights[0]) + \
                                       np.matmul(inputs, split_inputweights[0]) + split_recurrentbias[0])

            second_bit_output = np.tanh(np.matmul(split_activations[1], split_recurrentweights[1]) + \
                                       np.matmul(inputs, split_inputweights[1]) + split_recurrentbias[1])

            third_bit_output = np.tanh(np.matmul(split_activations[2], split_recurrentweights[2]) + \
                                       np.matmul(inputs, split_inputweights[2]) + split_recurrentbias[2])

            return [first_bit_output, second_bit_output, third_bit_output]

        def dense_layer(recurrent_activations):
            first_bit_weights = np.vstack(self.weights_by_number['first_bit']['weights'])
            second_bit_weights = np.vstack(self.weights_by_number['second_bit']['weights'])
            third_bit_weights = np.vstack(self.weights_by_number['third_bit']['weights'])

            first_bit_output = np.matmul(recurrent_activations[0], first_bit_weights)
            second_bit_output = np.matmul(recurrent_activations[1], second_bit_weights)
            third_bit_output = np.matmul(recurrent_activations[2], third_bit_weights)

            return np.hstack((first_bit_output, second_bit_output, third_bit_output))

        def pooling_layer(output_from_domains, pooling_weights):
            return np.matmul(output_from_domains, pooling_weights)


        return recurrent_layer, dense_layer, pooling_layer

    def get_recurrent_kernel_splits(self):
        first_bit_index = np.asarray(self.weights_by_number['first_bit']['index'])
        second_bit_index = np.asarray(self.weights_by_number['second_bit']['index'])
        third_bit_index = np.asarray(self.weights_by_number['third_bit']['index'])

        return [first_bit_index, second_bit_index, third_bit_index]

    def get_recurrent_subkernel(self, weights):
        inputweights, recurrentweights, recurrentbias = weights[0], weights[1], weights[2]
        list_of_indices = self.get_recurrent_kernel_splits()

        first_bit_inputweights = inputweights[:,list_of_indices[0]]
        first_bit_recurrentweights = recurrentweights[list_of_indices[0], :]
        first_bit_recurrentweights = first_bit_recurrentweights[:, list_of_indices[0]]
        first_bit_recurrentbias = recurrentbias[list_of_indices[0]]

        second_bit_inputweights = inputweights[:,list_of_indices[1]]
        second_bit_recurrentweights = recurrentweights[list_of_indices[1], :]
        second_bit_recurrentweights = second_bit_recurrentweights[:, list_of_indices[1]]
        second_bit_recurrentbias = recurrentbias[list_of_indices[1]]

        third_bit_inputweights = inputweights[:,list_of_indices[2]]
        third_bit_recurrentweights = recurrentweights[list_of_indices[2], :]
        third_bit_recurrentweights = third_bit_recurrentweights[:, list_of_indices[2]]
        third_bit_recurrentbias = recurrentbias[list_of_indices[2]]

        split_inputweights = [first_bit_inputweights, second_bit_inputweights, third_bit_inputweights]
        split_recurrentweights = [first_bit_recurrentweights, second_bit_recurrentweights, third_bit_recurrentweights]
        split_recurrentbias = [first_bit_recurrentbias, second_bit_recurrentbias, third_bit_recurrentbias]

        return split_inputweights, split_recurrentweights, split_recurrentbias

    def split_recurrent_activations(self, activations):
        list_of_indices = self.get_recurrent_kernel_splits()

        first_bit_activations = activations[:, list_of_indices[0]]
        second_bit_activations = activations[:, list_of_indices[1]]
        third_bit_activations = activations[:, list_of_indices[2]]

        split_activations = [first_bit_activations, second_bit_activations, third_bit_activations]

        return split_activations

    def reconstruct_domains(self, activations, outputs, bit):
        mean_neuron_activations = np.mean(activations, axis=0)

        # activations_first_number = np.zeros(activations.shape)
        activations_first_number = np.repeat(np.reshape(mean_neuron_activations, (-1, self.n_hidden)), activations.shape[0],
                                             axis=0)
        first_bit_neurons = np.asarray(self.weights_by_number[bit]['index'])
        activations_first_number[:, first_bit_neurons] = activations[:, first_bit_neurons]

        first_bit_neuron_weights = np.vstack(self.weights_by_number[bit]['weights'])

        firstbit_neuron_activations_reconstructed = np.matmul(outputs, first_bit_neuron_weights.transpose())
        activations_first_number[:, first_bit_neurons] = firstbit_neuron_activations_reconstructed

        return activations_first_number

    def serialize_recurrent_layer(self, weights):

        evals, evecs = np.linalg.eig(weights[1])
        diagonal_evals = np.real(np.diag(evals))
        real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)
        #reconstructed_matrices = []
        for i in range(len(weights[1])):

            #diagonal_evals = np.zeros((24, 24))
            #diagonal_evals[i, i] = evals[i]**(1/24)

            #reconstructed_weights = evecs @ diagonal_evals @ np.linalg.pinv(evecs)
            if img_parts[i] > 0:
                diagonal_evals[i, i + 1] = img_parts[i]
                diagonal_evals[i + 1, i] = img_parts[i + 1]
                evecs_c[:, i] = np.real(evecs[:, i])
                evecs_c[:, i + 1] = np.imag(evecs[:, i])
                i += 2


            #reconstructed_matrices.append(reconstructed_weights)

        return diagonal_evals



