import numpy as np
import multiprocessing as mp
import numdifftools as nd
from scipy.optimize import minimize
import sklearn.decomposition as skld
import matplotlib.pyplot as plt

class FixedPointFinder:
    def __init__(self, hps, data_hps):
        self.hps = hps
        self.data_hps = data_hps


    def parallel_minimization(self, weights, inputweights, activation, inputs, method):
        """Function to set up parallel processing of minimization for fixed points


        """
        pool = mp.Pool(mp.cpu_count())
        ids = np.arange(0, activation.shape[0])
        print(len(ids), " minimizations to parallelize.")
        x0, input = activation[ids, :], inputs[ids, :]
        combind = []
        for i in range(x0.shape[0]):
            combind.append((x0[i, :], input[i, :], weights, inputweights, method))

        if self.hps['rnn_type'] == 'vanilla':
            self.fixedpoints = pool.map(self._minimizRNN, combind, chunksize=1)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        self._handle_bad_approximations()
        return self.good_fixed_points


    def _minimizRNN(self, combined):
        n_hidden = self.hps['n_hidden']
        x0, input, weights, inputweights, method = combined[0], \
                                                   combined[1], \
                                                   combined[2], combined[3], \
                                                   combined[4]
        fun = lambda x: 0.5 * sum(
            (- x[0:n_hidden] + np.matmul(weights.transpose(), np.tanh(x[0:n_hidden])) + np.matmul(inputweights.transpose(), input)) ** 2)
        options = {'gtol': 1e-12, 'disp': True}
        jac, hes = nd.Gradient(fun), nd.Hessian(fun)
        y = fun(x0)
        print("First function evaluation:", y)
        fixed_point = minimize(fun, x0, method=method, jac=jac, hess=hes,
                                    options=options)
        return fixed_point

    def _handle_bad_approximations(self):
        """This functions identifies approximations where the minmization
        was not successful."""

        self.bad_fixed_points = []
        self.good_fixed_points = []
        for i in range(len(self.fixedpoints)):
            if not self.fixedpoints[i].success:
                self.bad_fixed_points.append(self.fixedpoints[i])
            else:
                self.good_fixed_points.append(self.fixedpoints[i])

    def plot_fixed_points(self, activations):

        self._extract_fixed_point_locations()

        pca = skld.PCA(3)
        pca.fit(activations[0, :, :])
        X_pca = pca.transform(activations[0, :, :])
        new_pca = pca.transform(self.fixed_point_locations)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                   marker='o', s=5)

        ax.scatter(new_pca[:, 0], new_pca[:, 1], new_pca[:, 2],
                   marker='x', s=30)
        plt.title('PCA')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()
# TODO: plotting function needs more modules and option to include or not include trajectories
# TODO: also needed: decision boundary for when a fixed point is considered unique and when not
    def _extract_fixed_point_locations(self):
        # processing of minimisation results for pca
        fixed_point_location = []
        for i in range(len(self.good_fixed_points)):
            fixed_point_location.append(self.good_fixed_points[i].x)
        self.fixed_point_locations = np.vstack(fixed_point_location)

# TODO: implement other architectures
# TODO: interpret Jacobian

