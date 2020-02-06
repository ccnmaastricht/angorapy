import numpy as np
import multiprocessing as mp
import numdifftools as nd
from scipy.optimize import minimize
import sklearn.decomposition as skld
import matplotlib.pyplot as plt
from LSTM.minimization import backproprnn, backpropgru
from mayavi import mlab


class FixedPointFinder:
    """The class FixedPointFinder creates a fixedpoint dictionary and has functionality to visualize RNN trajectories
    projected in 3D. A fixedpoint dictionary contain 'fun', the function evaluation at the fixedpoint 'x' and the
    corresponding 'jacobian'

    Args:
        hps: Dictionary of hyperparameters.

            unique_tol: Tolerance for when a fixedpoint will be considered unique, i.e. when two points are further
            away from each than the tolerance, they will be considered unique and discarded otherwise. Default: 1e-03.

            threshold: Minimization criteria. A fixedpoint must evaluate below the threshold in order to be considered
            a slow/fixed point. This value depends on the task of the RNN. Default for 3-Bit Flip-FLop: 1e-12.

            rnn_type: Specifying the architecture of the network. The network architecture defines the dynamical system.
            Must be one of ['vanilla', 'gru', 'lstm']. No default.

            n_hidden: Specifiying the number hidden units in the recurrent layer. No default.

            algorithm: Algorithm that shall be employed for the minimization. Must be one of: scipy, adam. It is recommended
            to use any of the two for vanilla architectures but adam for gru and lstm architectures. No default.

        weights: list of weights as returned by tensorflow.keras for recurrent layer. The list must contain three objects:
        input weights, recurrent weights and biases.

        inputs: Input to the recurrent layer. First two dimensions must match first two dimesnions of x0

        x0: acitvations of recurrent layer.

        method: methd to pass to minimize function from scipy. Default: Newton-CG. """
    def __init__(self, hps, weights, inputs, x0, method: str = "Newton-CG"):
        self.hps = hps
        self.unique_tol = hps['unique_tol']
        self.threshold = hps['threshold']
        self.minimization_distance = 10.0

        self.fixedpoints = []
        self.weights = weights
        if len(x0.shape) == 3:
            x, input = x0[0, :, :], inputs[0, :, :]
        else:
            x, input = x0, inputs

        if self.hps['algorithm'] == 'scipy':
            self.parallel_minimization(input, x, method)
            self.plot_fixed_points(x0, )
            self.plot_velocities(x0, )
        else:
            self.backprop(input, x)
            self.plot_fixed_points(x0, )
            self.plot_velocities(x0, )

    def backprop(self, inputs, x0):
        """Function to set up parallel processing of minimization for fixed points using adam as optimizer.

        Args:
            inputs: constant input at timestep of initial condition (IC). Object needs to have shape:
            [n_timesteps x n_hidden].
            x0: object containing ICs to begin optimization from. Object needs to have shape:
            [n_timesteps x n_hidden]."""
        weights, n_hidden, combind = self.weights, self.hps['n_hidden'], []
        combind = []
        for i in range(x0.shape[0]):  # prepare iterable object for parallelization
            combind.append((x0[i, :], inputs[i, :], weights, n_hidden))

        pool = mp.Pool(mp.cpu_count())
        if self.hps['rnn_type'] == 'vanilla':
            self.fixedpoints = pool.map(backproprnn, combind, chunksize=1)
        elif self.hps['rnn_type'] == 'gru':
            self.fixedpoints = pool.map(backpropgru, combind, chunksize=1)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        self._handle_bad_approximations(x0, inputs)

    def parallel_minimization(self, inputs, x0, method):
        """Function to set up parallel processing of minimization for fixed points using minimize from scipy.

        Args:
            inputs: constant input at timestep of initial condition (IC). Object needs to have shape:
            [n_timesteps x n_hidden].
            x0: object containing ICs to begin optimization from. Object needs to have shape:
            [n_timesteps x n_hidden].
            method: optimization method to be communicated to scipy.optimize.minimize. Default: "Newton-CG"

        """
        pool = mp.Pool(mp.cpu_count())
        print(x0.shape[0], " minimizations to parallelize.")
        weights, n_hidden, combind = self.weights, self.hps['n_hidden'], []

        for i in range(x0.shape[0]):  # prepare iterable object for parallelization
            combind.append((x0[i, :], inputs[i, :], weights, method, n_hidden))

        if self.hps['rnn_type'] == 'vanilla':
            self.fixedpoints = pool.map(self._minimizrnn, combind, chunksize=1)
        elif self.hps['rnn_type'] == 'gru':
            self.fixedpoints = pool.map(self._minimizgru, combind, chunksize=1)
        elif self.hps['rnn_type'] == 'lstm':
            self.fixedpoints = pool.map(self._minimizlstm, combind, chunksize=1)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        self._handle_bad_approximations(x0, inputs)

    @staticmethod
    def _minimizrnn(combined):
        x0, input, weights, method, n_hidden = combined[0], combined[1], combined[2], combined[3], combined[4]
        weights, inputweights, b = weights[1], weights[0], weights[2]
        projection_b = np.matmul(input, inputweights) + b
        fun = lambda x: 0.5 * sum((- x[0:n_hidden] + np.matmul(np.tanh(x[0:n_hidden]), weights) + b) ** 2)
        options = {'disp': True}
        jac, hes = nd.Gradient(fun), nd.Hessian(fun)
        y = fun(x0)
        print("First function evaluation:", y)
        fixed_point = minimize(fun, x0, method=method, jac=jac, hess=hes,
                               options=options)

        jac_fun = lambda x: - np.eye(n_hidden, n_hidden) + weights * (1 - np.tanh(x[0:n_hidden]) ** 2)
        fixed_point.jac = jac_fun(fixed_point.x)
        fixedpoint = {'fun': fixed_point.fun,
                      'x': fixed_point.x,
                      'jac': fixed_point.jac}
        return fixedpoint

    @staticmethod
    def _minimizgru(combined):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        x0, input, weights, method, n_hidden = combined[0], combined[1], combined[2], combined[3], combined[4]
        z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden), np.arange(2 * n_hidden, 3 * n_hidden)
        W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
        U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
        b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

        # z_projection_b = np.matmul(input, W_z) + b_z
        # r_projection_b = np.matmul(input, W_r) + b_r
        # g_projection_b = np.matmul(input, W_h) + b_h

        z_fun = lambda x: sigmoid(np.matmul(x[0:n_hidden], U_z) + b_z)
        r_fun = lambda x: sigmoid(np.matmul(x[0:n_hidden], U_r) + b_r)
        g_fun = lambda x: np.tanh((r_fun(x[0:n_hidden]) * np.matmul(x[0:n_hidden], U_h) + b_h))

        fun = lambda x: 0.5 * sum(((1 - z_fun(x[0:n_hidden])) * (g_fun(x[0:n_hidden]) - x[0:n_hidden])) ** 2)
        options = {'disp': True}
        jac, hes = nd.Gradient(fun), nd.Hessian(fun)
        y = fun(x0)
        print("First function evaluation:", y)
        fixed_point = minimize(fun, x0, method=method, jac=jac, hess=hes,
                               options=options)

        dynamical_system = lambda x: (1 - z_fun(x[0:n_hidden])) * (g_fun(x[0:n_hidden]) - x[0:n_hidden])
        jac_fun = nd.Jacobian(dynamical_system)
        fixed_point.jac = jac_fun(fixed_point.x)

        return fixed_point

    @staticmethod
    def _minimizlstm(combined):
        """There is not yet code to extract the lstm state tuple. Thus there would be h but no c vector. NOT YET WORKING"""
        x0, input, weights, method, n_hidden = combined[0], \
                                               combined[1], \
                                               combined[2], combined[3], combined[4]

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        W, U, b = weights[0], weights[1], weights[2]

        W_i, W_f, W_c, W_o = W[:, :n_hidden].transpose(), W[:, n_hidden:2 * n_hidden].transpose(), \
                             W[:, 2 * n_hidden:3 * n_hidden].transpose(), W[:, 3 * n_hidden:4 * n_hidden].transpose()
        U_i, U_f, U_c, U_o = U[:, :n_hidden].transpose(), U[:, n_hidden:2 * n_hidden].transpose(), \
                             U[:, 2 * n_hidden:3 * n_hidden].transpose(), U[:, 3 * n_hidden:4 * n_hidden].transpose()
        b_i, b_f, b_c, b_o = b[0, :n_hidden].transpose(), b[0, n_hidden:2 * n_hidden].transpose(), \
                             b[0, 2 * n_hidden:3 * n_hidden].transpose(), b[0, 3 * n_hidden:4 * n_hidden].transpose()
        f_fun = lambda x: sigmoid(np.matmul(W_f, input) + np.matmul(U_f, x[0:n_hidden]) + b_f)
        i_fun = lambda x: sigmoid(np.matmul(W_i, input) + np.matmul(U_i, x[0:n_hidden]) + b_i)
        o_fun = lambda x: sigmoid(np.matmul(W_o, input) + np.matmul(U_o, x[0:n_hidden]) + b_o)
        c_fun = lambda x, c: f_fun(x[0:n_hidden]) * c[0:n_hidden] + i_fun(x[0:n_hidden]) * \
                             np.tanh((np.matmul(W_c, input) + np.matmul(U_c, x[0:n_hidden]) + b_c) - c[0:n_hidden])
        # perhaps put h and c in as one object to minimize and split up in functions
        fun = lambda x, c: 0.5 * sum((o_fun(x[0:n_hidden]) * np.tanh(c_fun(x[0:n_hidden], c)) - x[0:n_hidden]) ** 2)

        options = {'disp': True}
        jac, hes = nd.Gradient(fun), nd.Hessian(fun)
        y = fun(x0)
        print("First function evaluation:", y)
        fixed_point = minimize(fun, x0, method=method, jac=jac, hess=hes,
                               options=options)

        dynamical_system = lambda x, c: o_fun(x[0:n_hidden]) * np.tanh(c_fun(x[0:n_hidden], c)) - x[0:n_hidden]
        jac_fun = nd.Jacobian(dynamical_system)
        fixed_point.jac = jac_fun(fixed_point.x)

        return fixed_point

    def _handle_bad_approximations(self, activation, inputs):
        """This functions identifies approximations where the minmization
        was
         a) not successful
         b) the fixed point moved further away from IC than minimization distance threshold
         c) minimization did not return q(x) < threshold"""

        self.bad_fixed_points = []
        self.good_fixed_points = []
        self.good_inputs = []
        for i in range(len(self.fixedpoints)):
 #           if np.sqrt(((activation[i, :] - self.fixedpoints[i]['x']) ** 2).sum()) > self.minimization_distance:
  #              self.bad_fixed_points.append(self.fixedpoints[i])
            if self.fixedpoints[i]['fun'] > self.threshold:
                self.bad_fixed_points.append(self.fixedpoints[i])
            else:
                self.good_fixed_points.append(self.fixedpoints[i])
                self.good_inputs.append(inputs[i, :])

    def plot_velocities(self, activations):
        """Function to evaluate and visualize velocities at all recorded activations of the recurrent layer."""
        if self.hps['rnn_type'] == 'vanilla':
            recurrentweights = self.weights[1]
            def func(x):
                return 0.5 * np.sum(((- x + np.matmul(np.tanh(x), recurrentweights))**2), axis=1)
        elif self.hps['rnn_type'] == 'gru':
            weights, n_hidden = self.weights, self.hps['n_hidden']
            z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden), np.arange(2 * n_hidden, 3 * n_hidden)
            W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
            U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
            b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

            #z_projection_b = np.matmul(input, W_z) + b_z
            #r_projection_b = np.matmul(input, W_r) + b_r
            #g_projection_b = np.matmul(input, W_h) + b_h

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))


            func = lambda x: 0.5 * np.sum((((1 - sigmoid(np.matmul(x, U_z) + b_z)) * (np.tanh((sigmoid(np.matmul(x, U_r) + b_r)  * np.matmul(x, U_h) + b_h)) - x)) ** 2), axis = 1)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        activations = np.vstack(activations)
        # get velocity at point
        vel = func(activations)

        pca = skld.PCA(3)
        pca.fit(activations)
        X_pca = pca.transform(activations)

        n_points = 3000
        mlab.plot3d(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
                    vel[:n_points])
        mlab.colorbar(orientation='vertical')
        mlab.show()

    def plot_fixed_points(self, activations, fixedpoints=None):
        if fixedpoints is not None:
            self.good_fixed_points = fixedpoints
        self._extract_fixed_point_locations()
        self._find_unique_fixed_points()
        self.unique_jac = [self.good_fixed_points[i] for i in self.unique_idx]



        activations = np.vstack(activations)

        pca = skld.PCA(3)
        pca.fit(activations)
        X_pca = pca.transform(activations)
        new_pca = pca.transform(self.unique_fixed_points)

        # somehow this block of code does not return values if put in function
        x_modes = []
        x_directions = []
        scale = 4
        for n in range(len(self.unique_jac)):

            trace = np.matrix.trace(self.unique_jac[n]['jac'])
            det = np.linalg.det(self.unique_jac[n]['jac'])

            if det < 0:
                print('saddle_point')
                x_modes.append(True)
                e_val, e_vecs = np.linalg.eig(self.unique_jac[n]['jac'])
                ids = np.argwhere(np.real(e_val) > 0)
                for i in range(len(ids)):
                    x_plus = self.unique_jac[n]['x'] + scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                    x_minus = self.unique_jac[n]['x'] - scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                    x_direction = np.vstack((x_plus, self.unique_jac[n]['x'], x_minus))
                    x_directions.append(np.real(x_direction))
            elif det > 0:
                if trace ** 2 - 4 * det > 0 and trace < 0:
                    print('stable fixed point was found.')
                    x_modes.append(False)
                    e_val, e_vecs = np.linalg.eig(self.unique_jac[n]['jac'])
                    ids = np.argwhere(np.real(e_val) > 0)
                    for i in range(len(ids)):
                        x_plus = self.unique_jac[n]['x'] + scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                        x_minus = self.unique_jac[n]['x'] - scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                        x_direction = np.vstack((x_plus, self.unique_jac[n]['x'], x_minus))
                        x_directions.append(np.real(x_direction))
                elif trace ** 2 - 4 * det > 0 and trace > 0:
                    print('unstable fixed point was found')
                else:
                    print('center was found.')
                    x_modes.append(False)
            else:
                print('fixed point manifold was found.')

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(X_pca[:5000, 0], X_pca[:5000, 1], X_pca[:5000, 2],
                linewidth=0.2)
        for i in range(len(x_modes)):
            if not x_modes[i]:
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='k')
            else:
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='r')
                for p in range(len(x_directions)):
                    direction_matrix = pca.transform(x_directions[p])
                    ax.plot(direction_matrix[:, 0], direction_matrix[:, 1], direction_matrix[:, 2],
                            c='r', linewidth=0.8)

        plt.title('PCA using modeltype: ' + self.hps['rnn_type'])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()

    def _extract_fixed_point_locations(self):
        """Processing of minimisation results for pca. The function takes one fixedpoint object at a time and
        puts all coordinates in single array."""
        fixed_point_location = []
        for i in range(len(self.good_fixed_points)):
            fixed_point_location.append(self.good_fixed_points[i]['x'])
        self.fixed_point_locations = np.vstack(fixed_point_location)

    def _find_unique_fixed_points(self):
        """Identify fixed points that are unique within a distance threshold
        of """
        d = int(np.round(np.max([0 - np.log10(self.unique_tol)])))
        ux, idx = np.unique(self.fixed_point_locations.round(decimals=d), axis=0, return_index=True)

        self.unique_fixed_points = ux
        self.unique_idx = idx
