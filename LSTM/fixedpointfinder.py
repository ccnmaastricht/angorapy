import numpy as np
import multiprocessing as mp
import numdifftools as nd
from scipy.optimize import minimize
import sklearn.decomposition as skld
import matplotlib.pyplot as plt
from LSTM.minimization import backproprnn, backpropgru
from mayavi import mlab
from LSTM.build_utils import build_rnn_ds, build_gru_ds, build_lstm_ds


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

            n_hidden: Specifiying the number of hidden units in the recurrent layer. No default.

            algorithm: Algorithm that shall be employed for the minimization. Must be one of: scipy, adam. It is recommended
            to use any of the two for vanilla architectures but adam for gru and lstm architectures. No default.

            n_points: Number of points to use to plot the trajectories the network took. Recommended 200-5000, otherwise
            the plot will look too sparse or too crowded. Default: 3000.

            use_input: boolean parameter indicating if input to the recurrent layer shall be used during minimization or
            not. Default: False

        scipy_hps: Dictionary of hyperparameters specifically for minimize from scipy.

            method: Method to employ for minimization using the scipy package. Default: "Newton-CG".

            display: boolean array indication, if information about the minimization shall be printed to the console

        adam_hps: Dictionary of hyperparameters specifically for adam optimizer.

            max_iter: maximum number of iterations to run backpropagation for. Default: 5000.



        weights: list of weights as returned by tensorflow.keras for recurrent layer. The list must contain three objects:
        input weights, recurrent weights and biases.

        inputs: Input to the recurrent layer. First two dimensions must match first two dimensions of x0

        x0: activations of recurrent layer."""
    def __init__(self, hps, weights, inputs, x0):
        self.hps = hps
        self.unique_tol = hps['unique_tol']
        self.threshold = hps['threshold']
        self.minimization_distance = 10.0
        self.n_points = 3000
        self.use_input = False
        self.n_ic = hps['n_ic']
        self.scipy_hps = {'method': hps['method'],
                          'display': hps['display']}

        self.adam_hps = {'max_iter': hps['max_iter'],
                         'n_init': hps['n_init'],
                         'lr': hps['lr'],
                         'gradientnormclip': hps['gradientnormclip'],
                         'print_every': hps['print_every']}

        self._print_hps()
        self.fixedpoints = []
        self.weights = weights
        if len(x0.shape) == 3:
            x, input = x0[0, :, :], inputs[0, :, :]
        else:
            x, input = x0, inputs

        if self.hps['algorithm'] == 'scipy':
            self.parallel_minimization(input, x)
            self.plot_fixed_points(x0, )
            self.plot_velocities(x0, )
        elif self.hps['algorithm'] == 'adam':
            self.backprop(input, x)
            self.plot_fixed_points(x0, )
            self.plot_velocities(x0, )
        else:
            self.exponential_minimization(inputs, x0)
            # self.plot_fixed_points(x0, )
            # self.plot_velocities(x0, )

# TODO: add hyperparameters:

    def backprop(self, inputs, x0):
        """Function to set up parallel processing of minimization for fixed points using adam as optimizer.

        Args:
            inputs: constant input at timestep of initial condition (IC). Object needs to have shape:
            [n_timesteps x n_hidden].
            x0: object containing ICs to begin optimization from. Object needs to have shape:
            [n_timesteps x n_hidden]."""
        weights, n_hidden, combind = self.weights, self.hps['n_hidden'], []
        combind = []
        for i in range(self.n_ic-10):  # prepare iterable object for parallelization
            combind.append((x0[i:i+self.adam_hps['n_init'], :], inputs[i:i+self.adam_hps['n_init'], :],
                            weights, n_hidden, self.adam_hps, self.use_input))
            i += self.adam_hps['n_init']

        pool = mp.Pool(mp.cpu_count())
        if self.hps['rnn_type'] == 'vanilla':
            self.fixedpoints = pool.map(backproprnn, combind, chunksize=1)
        elif self.hps['rnn_type'] == 'gru':
            self.fixedpoints = pool.map(backpropgru, combind, chunksize=1)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])
        self.fixedpoints = [item for sublist in self.fixedpoints for item in sublist]
        self._handle_bad_approximations(x0, inputs)

    def parallel_minimization(self, inputs, x0):
        """Function to set up parallel processing of minimization for fixed points using minimize from scipy.

        Args:
            inputs: constant input at timestep of initial condition (IC). Object needs to have shape:
            [n_timesteps x n_hidden].
            x0: object containing ICs to begin optimization from. Object needs to have shape:
            [n_timesteps x n_hidden].
            method: optimization method to be communicated to scipy.optimize.minimize. Default: "Newton-CG"

        """
        pool = mp.Pool(mp.cpu_count())
        print(self.n_ic, " minimizations to parallelize.")
        weights, n_hidden, combind = self.weights, self.hps['n_hidden'], []

        for i in range(self.n_ic):  # prepare iterable object for parallelization
            combind.append((x0[i, :], inputs[i, :], weights, self.scipy_hps, n_hidden, self.use_input))

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

    def exponential_minimization(self, inputs, x0):
        x0 = np.vstack(x0)

        pca = skld.PCA(3)
        pca.fit(x0)
        X_pca = pca.transform(x0)

        minimum, maximum = np.amin(X_pca, axis=0), np.amax(X_pca, axis=0)

        n_samples = 50
        self.search_points = np.linspace(minimum, maximum, n_samples)

        self.x, self.y, self.z = np.meshgrid(self.search_points[:, 0], self.search_points[:, 1], self.search_points[:, 2],
                                             indexing='ij')
        self.positions = np.vstack([self.x.ravel(), self.y.ravel(), self.z.ravel()]).transpose()
        self.original_space = pca.inverse_transform(self.positions)

        fun, _ = build_gru_ds(self.weights, self.hps['n_hidden'], inputs, use_input=False)

        self.value = fun(self.original_space)

        k = 10000
        idx = np.argpartition(self.value, k)
        self.new_positions = self.positions[idx[:k]]
        self.new_value = self.value[idx[:k]]
        n_points = k
        mlab.plot3d(self.new_positions[:n_points, 0], self.new_positions[:n_points, 1], self.new_positions[:n_points, 2],
                    self.new_value[:n_points])
        mlab.colorbar(orientation='vertical')
        mlab.show()



    @staticmethod
    def _minimizrnn(combined):
        x0, input, weights, scipy_hps, n_hidden, use_input = combined[0], combined[1], combined[2], combined[3], \
                                                          combined[4], combined[5]


        fun = build_rnn_ds(weights, input, use_input=use_input)
        jac, hes = nd.Gradient(fun), nd.Hessian(fun)

        print("First function evaluation:", fun(x0))
        options = {'disp': scipy_hps['display']}
        fixed_point = minimize(fun, x0, method=scipy_hps['method'], jac=jac, hess=hes,
                               options=options)

        jac_fun = lambda x: - np.eye(n_hidden, n_hidden) + weights * (1 - np.tanh(x) ** 2)
        fixed_point.jac = jac_fun(fixed_point.x)
        fixedpoint = {'fun': fixed_point.fun,
                      'x': fixed_point.x,
                      'jac': fixed_point.jac}
        return fixedpoint

    @staticmethod
    def _minimizgru(combined):
        x0, input, weights, scipy_hps, n_hidden, use_input = combined[0], combined[1], combined[2], combined[3], \
                                                          combined[4], combined[5]

        fun, dynamical_system = build_gru_ds(weights, input, n_hidden, use_input=use_input)

        jac, hes = nd.Gradient(fun), nd.Hessian(fun)

        print("First function evaluation:", fun(x0))
        options = {'disp': scipy_hps['display']}
        fixed_point = minimize(fun, x0, method=scipy_hps['method'], jac=jac, hess=hes,
                               options=options)

        jac_fun = nd.Jacobian(dynamical_system)
        fixed_point.jac = jac_fun(fixed_point.x)
        fixedpoint = {'fun': fixed_point.fun,
                      'x': fixed_point.x,
                      'jac': fixed_point.jac}

        return fixedpoint

    @staticmethod
    def _minimizlstm(combined):
        """There is not yet code to extract the lstm state tuple. Thus there would be h but no c vector. NOT YET WORKING"""
        x0, input, weights, method, n_hidden, use_input = combined[0], \
                                               combined[1], \
                                               combined[2], combined[3], combined[4], combined[5]

        fun, dynamical_system = build_lstm_ds(weights, input, n_hidden, use_input=use_input)

        jac, hes = nd.Gradient(fun), nd.Hessian(fun)

        print("First function evaluation:", fun(x0))
        options = {'disp': True}
        fixed_point = minimize(fun, x0, method=method, jac=jac, hess=hes,
                               options=options)

        jac_fun = nd.Jacobian(dynamical_system)
        fixed_point.jac = jac_fun(fixed_point.x)
        fixedpoint = {'fun': fixed_point.fun,
                      'x': fixed_point.x,
                      'jac': fixed_point.jac}

        return fixedpoint

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
               # self.good_inputs.append(inputs[i, :])

    def plot_velocities(self, activations):
        """Function to evaluate and visualize velocities at all recorded activations of the recurrent layer."""
        if self.hps['rnn_type'] == 'vanilla':
            func = build_rnn_ds(self.weights)
        elif self.hps['rnn_type'] == 'gru':
            weights, n_hidden = self.weights, self.hps['n_hidden']
            func, _ = build_gru_ds(weights, n_hidden)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        activations = np.vstack(activations)
        # get velocity at point
        vel = func(activations)

        pca = skld.PCA(3)
        pca.fit(activations)
        X_pca = pca.transform(activations)

        n_points = self.n_points
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
        scale = 2
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
        ax.plot(X_pca[:self.n_points, 0], X_pca[:self.n_points, 1], X_pca[:self.n_points, 2],
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

    def _print_hps(self):
        COLORS = dict(
            HEADER='\033[95m',
            OKBLUE='\033[94m',
            OKGREEN='\033[92m',
            WARNING='\033[93m',
            FAIL='\033[91m',
            ENDC='\033[0m',
            BOLD='\033[1m',
            UNDERLINE='\033[4m'
        )
        bc, ec, wn = COLORS["HEADER"], COLORS["ENDC"], COLORS["WARNING"]
        print(f"-----------------------------------------\n"
              f"{wn}Architecture to analyse {ec}: {bc}{self.hps['rnn_type']}{ec}\n"
              f"The layer has {bc}{self.hps['n_hidden']}{ec} recurrent units. \n"
              f"Using {bc}{self.hps['algorithm']}{ec} for minimization.\n"
              f"-----------------------------------------\n"
              f"{wn}HyperParameters{ec}: threshold - {self.hps['threshold']}\n unique_tolerance - {self.hps['unique_tol']}\n"
              f"-----------------------------------------\n")

