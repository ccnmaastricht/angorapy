import numpy as np
import multiprocessing as mp
import math
import numdifftools as nd
from scipy.optimize import minimize
import sklearn.decomposition as skld
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from utilities.model_management import build_sub_model_to
import tensorflow as tf


class FixedPointFinder:
    def __init__(self, hps, data_hps, model):
        self.hps = hps
        self.data_hps = data_hps
        self.unique_tol = 1e-03
        self.abundance_threshold = 0.04
        self.weights = model.get_layer(self.hps['rnn_type']).get_weights()
        self.model = model

    def parallel_minimization(self, inputs, activation, method):
        """Function to set up parallel processing of minimization for fixed points


        """
        pool = mp.Pool(mp.cpu_count())
        print(activation.shape[0], " minimizations to parallelize.")
        x0 = activation
        combind = []
        for i in range(x0.shape[0]):
            combind.append((x0[i, :], inputs[i, :], method))

        if self.hps['rnn_type'] == 'vanilla':
            self.fixedpoints = pool.map(self._minimizrnn, combind, chunksize=1)
        elif self.hps['rnn_type'] == 'gru':
            self.fixedpoints = pool.map(self._minimizgru, combind, chunksize=1)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        self._handle_bad_approximations()
        return self.good_fixed_points

    def _minimizrnn(self, combined):
        n_hidden = self.hps['n_hidden']
        x0, input, weights, method = combined[0], \
                                                   combined[1], \
                                                   self.weights, combined[2]
        weights, inputweights, b = weights[1], weights[0], weights[2]
        fun = lambda x: 0.5 * sum(
            (- x[0:n_hidden] + np.matmul(weights.transpose(), np.tanh(x[0:n_hidden])) +
             np.matmul(inputweights.transpose(), input) + b) ** 2)
        options = {'gtol': 1e-14, 'disp': True}
        jac, hes = nd.Gradient(fun), nd.Hessian(fun)
        y = fun(x0)
        print("First function evaluation:", y)
        fixed_point = minimize(fun, x0, method=method, jac=jac, hess=hes,
                                    options=options)
        return fixed_point

    def _minimizgru(self, combined):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        n_hidden = self.hps['n_hidden']
        x0, input, weights, method = combined[0], \
                                     combined[1], \
                                     self.weights, combined[2]
        z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2*n_hidden), np.arange(2*n_hidden, 3*n_hidden)
        W_z, W_r, W_h = weights[0][:, z].transpose(), weights[0][:, r].transpose(), weights[0][:, h].transpose()
        U_z, U_r, U_h = weights[1][:, z].transpose(), weights[1][:, r].transpose(), weights[1][:, h].transpose()
        b_z, b_r, b_h = weights[2][0, z].transpose(), weights[2][0, r].transpose(), weights[2][0, h].transpose()

        fun = lambda x: 0.5* sum(((1-sigmoid(np.matmul(W_z, input) + np.matmul(U_z, x[0:n_hidden]) + b_z)) @
                        np.tanh(np.matmul(W_h, input) + np.matmul(U_h, (sigmoid(np.matmul(W_r, input) +
                        np.matmul(U_r, x[0:n_hidden]) + b_r))) @ x[0:n_hidden] + b_h) - x[0:n_hidden])**2)
        options = {'gtol': 1e-14, 'disp': True}
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

    def plot_fixed_points(self, activations, fixedpoints = None):
        if fixedpoints is not None:
            self.good_fixed_points = fixedpoints
        self._extract_fixed_point_locations()
        self._find_unique_fixed_points()

        activations = np.vstack(activations)

        pca = skld.PCA(3)
        pca.fit(activations)
        X_pca = pca.transform(activations)
        new_pca = pca.transform(self.unique_fixed_points)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                linewidth=0.2)

        ax.scatter(new_pca[:, 0], new_pca[:, 1], new_pca[:, 2],
                   marker='x', s=30)
        plt.title('PCA')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()

    def _extract_fixed_point_locations(self):
        # processing of minimisation results for pca
        fixed_point_location = []
        for i in range(len(self.good_fixed_points)):
            fixed_point_location.append(self.good_fixed_points[i].x)
        self.fixed_point_locations = np.vstack(fixed_point_location)

    def _find_unique_fixed_points(self):
        candidates = squareform(pdist(self.fixed_point_locations, )) <= self.unique_tol
        self.unique_fixed_points = \
            self.fixed_point_locations[np.mean(candidates, axis=1) >= self.abundance_threshold, :]


# TODO: implement other architectures
# TODO: interpret Jacobian

