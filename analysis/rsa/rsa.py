from scipy.spatial.distance import pdist, squareform
import numpy as np
import itertools as it
from scipy.stats import pearsonr

class RSA:
    """Class to perform representational similarity analysis.

    This class can be initialized with activation data."""

    def __init__(self, recurrent_activations):
        self.recurrent_activations = recurrent_activations

    def compare_cross_neurons(self):
        pass

    def compare_cross_timeintervals(self, interval_size):
        """Window over time intervals and compare each time interval with all others

        Compute 1-correlation and put it into square matrix"""
        def moving_window(x, length):
            return [x[i: i + length, :] for i in range(0, (len(x) + 1 - length), length)]

        windowed_time = moving_window(self.recurrent_activations, interval_size)

        correlations = []
        for a, b in it.product(windowed_time, repeat=2):
            corr = np.empty(interval_size)
            for i in range(interval_size):
                corr[i] = pearsonr(a[i, :], b[i, :])
            correlations.append(np.mean(corr))

        n_intervals = int(np.sqrt(len(correlations)))
        similarity_matrix = np.reshape(np.vstack(correlations), (n_intervals, n_intervals))
        rdm = 1 - similarity_matrix

        return rdm
    def compare_input_to_recurrent_layer(self):
        pass

    def compare_input_cross_actions(self):
        pass

if __name__ == "__main__":
    pass