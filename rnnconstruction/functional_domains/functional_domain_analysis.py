import autograd.numpy as np

class FDA:

    def __init__(self, output_weights, classification_threshold):
        self.weights_by_number = self._classify_neurons(output_weights, classification_threshold)


    def _classify_neurons(self, output_weights, threshold):
        weights_by_number = {'number_one': {'weights': [], 'index': []},
                             'number_two': {'weights': [], 'index': []},
                             'number_three': {'weights': [], 'index': []}}
        k = 0

        for weights_per_neuron in output_weights:
            big_weights = np.abs(weights_per_neuron)
            biggest_weight = np.max(big_weights)
            difference = biggest_weight - big_weights
            ratio_biggest_to_weights = difference / biggest_weight
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
            first_bit_weights = np.vstack(self.weights_by_number['number_one']['weights'])
            second_bit_weights = np.vstack(self.weights_by_number['number_two']['weights'])
            third_bit_weights = np.vstack(self.weights_by_number['number_two']['weights'])

            first_bit_output = np.matmul(recurrent_activations[0], first_bit_weights)
            second_bit_output = np.matmul(recurrent_activations[1], second_bit_weights)
            third_bit_output = np.matmul(recurrent_activations[2], third_bit_weights)

            return np.hstack((first_bit_output, second_bit_output, third_bit_output))

        def pooling_layer(output_from_domains, pooling_weights):
            return np.matmul(output_from_domains, pooling_weights)

        return recurrent_layer, dense_layer, pooling_layer

    def get_recurrent_kernel_splits(self):
        first_bit_index = np.asarray(self.weights_by_number['number_one']['index'])
        second_bit_index = np.asarray(self.weights_by_number['number_two']['index'])
        third_bit_index = np.asarray(self.weights_by_number['number_two']['index'])

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