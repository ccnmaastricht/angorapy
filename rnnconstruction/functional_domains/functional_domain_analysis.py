import autograd.numpy as np

class FDA:

    def __init__(self):
        pass

    def classify_neurons(self, output_weights, threshold):
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

    @staticmethod
    def reconstruct_model_with_domains(weights, weights_by_number):

        def split_recurrent_kernel(weights_by_number):
            first_bit_index = np.asarray(weights_by_number['number_one']['index'])
            second_bit_index = np.asarray(weights_by_number['number_two']['index'])
            third_bit_index = np.asarray(weights_by_number['number_two']['index'])

            return [first_bit_index, second_bit_index, third_bit_index]

        def recurrent_layer(inputs, activations):
            inputweights, recurrentweights, recurrentbias = weights[0], weights[1], weights[2]
            list_of_indices = split_recurrent_kernel(weights_by_number)

            first_bit_inputweights, first_bit_recurrentweights, first_bit_recurrentbias = inputweights[:,
                                                                                          list_of_indices[0]], \
                                                                                          recurrentweights[
                                                                                          list_of_indices[0], :], \
                                                                                          recurrentbias[
                                                                                              list_of_indices[0]]
            first_bit_recurrentweights = first_bit_recurrentweights[:, list_of_indices[0]]

            second_bit_inputweights, second_bit_recurrentweights, second_bit_recurrentbias = inputweights[:,
                                                                                             list_of_indices[1]], \
                                                                                             recurrentweights[
                                                                                             list_of_indices[1], :], \
                                                                                             recurrentbias[
                                                                                                 list_of_indices[1]]
            second_bit_recurrentweights = second_bit_recurrentweights[:, list_of_indices[1]]

            third_bit_inputweights, third_bit_recurrentweights, third_bit_recurrentbias = inputweights[:,
                                                                                          list_of_indices[2]], \
                                                                                          recurrentweights[
                                                                                          list_of_indices[2], :], \
                                                                                          recurrentbias[
                                                                                              list_of_indices[2]]
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

            return np.hstack((first_bit_output, second_bit_output, third_bit_output))

        def pooling_layer(output_from_domains, pooling_weights):
            return np.matmul(output_from_domains, pooling_weights)

        return recurrent_layer, dense_layer, pooling_layer