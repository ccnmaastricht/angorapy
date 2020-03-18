from typing import Optional, Any

import numpy as np
import multiprocessing as mp
import numdifftools as nd
from scipy.optimize import minimize
from fixedpointfinder.build_utils import build_rnn_ds, build_gru_ds, build_lstm_ds
from fixedpointfinder.minimization import adam_optimizer, adam_lstm
from utilities.model_utils import build_sub_model_to
import tensorflow as tf


class FixedPointFinder(object):

    _default_hps = {'q_threshold': 1e-12,
                    'tol_unique': 1e-03,
                    'verbose': True,
                    'random_seed': 0}


    def __init__(self, weights, rnn_type,
                 q_threshold=_default_hps['q_threshold'],
                 tol_unique=_default_hps['tol_unique'],
                 verbose=_default_hps['verbose'],
                 random_seed=_default_hps['random_seed']):
        """The class FixedPointFinder creates a fixedpoint dictionary. A fixedpoint dictionary contain 'fun',
        the function evaluation at the fixedpoint 'x' and the corresponding 'jacobian'.

        Args:
            tol_unique: Tolerance for when a fixedpoint will be considered unique, i.e. when two points are further
            away from each than the tolerance, they will be considered unique and discarded otherwise. Default: 1e-03.

            threshold: Minimization criteria. A fixedpoint must evaluate below the threshold in order to be considered
            a slow/fixed point. This value depends on the task of the RNN. Default for 3-Bit Flip-FLop: 1e-12.

            rnn_type: Specifying the architecture of the network. The network architecture defines the dynamical system.
            Must be one of ['vanilla', 'gru', 'lstm']. No default.

            verbose: Boolean indicating if fixedpointfinder should act verbose. Default: True.

            random_seed: Random seed to make sampling of states reproducible. Default: 0"""

        self.weights = weights
        self.rnn_type = rnn_type

        self.q_threshold = q_threshold
        self.unique_tol = tol_unique

        self.verbose = verbose

        self.rng = np.random.RandomState(random_seed)

        if self.rnn_type == 'vanilla':
            self.n_hidden = int(weights[1].shape[1])
        elif self.rnn_type == 'gru':
            self.n_hidden = int(weights[1].shape[1] / 3)
        elif self.rnn_type == 'lstm':
            self.n_hidden = int(weights[1].shape[1] / 4)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.rnn_type)

        if self.verbose:
            self._print_hps()

    def sample_inputs_and_states(self, activations, inputs, n_inits, noise_level):
        """Draws [n_inits] random samples from recurrent layer activations.

        Args:
            activations: numpy array containing input dependent activations of
            the recurrent layer. Can have either shape [n_batches x n_time x n_units]
            or [n_time x n_units]. no default.

            inputs: numpy array containing inputs corresponding to activations
            of recurrent layer. Can have either shape [n_batches x n_time x n_units]
            or [n_time x n_units]. no default.

            n_inits: integer specifying how many samples shall be drawn from the
            numpy array.

            noise_level: non-negative integer specifying gaussian radius of noise added
            to the samples.

        Returns:
            sampled_activations: numpy array of sampled activations. Will have dimensions
            [n_init x n_units].

            sampled_inputs: numpy array of sampled inputs. Will have dimensions
            [n_init x n_units]."""
        if self.rnn_type == 'lstm':
            activations = np.hstack((activations[1], activations[2]))
            inputs = np.concatenate(inputs, axis=0)
        if len(activations.shape) == 3:
            activations = np.vstack(activations)
            inputs = np.vstack(inputs)

        init_idx = self.rng.randint(activations.shape[0], size=n_inits)

        sampled_activations = activations[init_idx, :]
        sampled_inputs = inputs[init_idx, :]

        sampled_activations = self._add_gaussian_noise(sampled_activations, noise_level)

        return sampled_activations, sampled_inputs


    def sample_states(self, activations, n_inits, noise_level=0.5):
        """Draws [n_inits] random samples from recurrent layer activations.

        Args:
            activations: numpy array containing input dependent activations of
            the recurrent layer. Can have either shape [n_batches x n_time x n_units]
            or [n_time x n_units]. no default.

            n_inits: integer specifying how many samples shall be drawn from the
            numpy array.

            noise_level: non-negative integer specifying gaussian radius of noise added
            to the samples.

        Returns:
            sampled_activations: numpy array of sampled activations. Will have dimensions
            [n_init x n_units]."""
        if self.rnn_type == 'lstm':
            activations = np.hstack(activations[1:])

        if len(activations.shape) == 3:
            activations = np.vstack(activations)

        init_idx = self.rng.randint(activations.shape[0], size=n_inits)

        sampled_activations = activations[init_idx, :]

        sampled_activations = self._add_gaussian_noise(sampled_activations, noise_level)

        return sampled_activations

    def _add_gaussian_noise(self, data, noise_scale=0.0):
        """ Adds IID Gaussian noise to Numpy data.
        Args:
            data: Numpy array.
            noise_scale: (Optional) non-negative scalar indicating the
            standard deviation of the Gaussian noise samples to be generated.
            Default: 0.0.
        Returns:
            Numpy array with shape matching that of data.
        Raises:
            ValueError if noise_scale is negative.
        """

        # Add IID Gaussian noise
        if noise_scale == 0.0:
            return data # no noise to add
        if noise_scale > 0.0:
            return data + noise_scale * self.rng.randn(*data.shape)
        elif noise_scale < 0.0:
            raise ValueError('noise_scale must be non-negative,'
                             ' but was %f' % noise_scale)

    def _handle_bad_approximations(self, fps):
        """This functions identifies approximations where the minmization
        was
         a) the fixed point moved further away from IC than minimization distance threshold
         b) minimization did not return q(x) < q_threshold"""

        bad_fixed_points = []
        good_fixed_points = []
        for fp in fps:
            if fp['fun'] > self.q_threshold:
                bad_fixed_points.append(fp)
            else:
                good_fixed_points.append(fp)

        return good_fixed_points, bad_fixed_points

    def compute_velocities(self, activations, input):
        """Function to evaluate velocities at all recorded activations of the recurrent layer.

        Args:
             """
        if len(activations.shape) == 3:
            activations = np.vstack(activations)
            input = np.vstack(input)

        if self.rnn_type == 'vanilla':
            func, _ = build_rnn_ds(self.weights, self.n_hidden, input, 'velocity')
        elif self.rnn_type == 'gru':
            func, _ = build_gru_ds(self.weights, self.n_hidden, input, 'velocity')
        elif self.rnn_type == 'lstm':
            func, _ = build_lstm_ds(self.weights, input, self.n_hidden, 'velocity')
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.rnn_type)

        # get velocity at point
        velocities = func(activations)
        return velocities

    def _find_unique_fixed_points(self, fps):
        """Identify fixed points that are unique within a distance threshold
        of """

        def extract_fixed_point_locations(fps):
            """Processing of minimisation results for pca. The function takes one fixedpoint object at a time and
            puts all coordinates in single array."""
            fixed_point_location = [fp['x'] for fp in fps]
            try:
                fixed_point_locations = np.vstack(fixed_point_location)
            except:
                raise ValueError('No fixed points were found with the current settings. \n'
                                 'It is recommended to rerun after adjusting one or more of the'
                                 'following settings: \n'
                                 '1. use more samples. \n'
                                 '2. adjust q_threshold to larger values. \n'
                                 'If you are using adam, try as well:\n'
                                 '1. adjust learning rate: if optimization seemed unstable, try smaller. If '
                                 'learning was simply very slow, try larger. \n'
                                 '2. increase maximum number of iterations.\n'
                                 '3. adjust hps for adaptive learning rate and/or adaptive gradient norm clip.')

            return fixed_point_locations

        fps_locations = extract_fixed_point_locations(fps)
        d = int(np.round(np.max([0 - np.log10(self.unique_tol)])))
        ux, idx = np.unique(fps_locations.round(decimals=d), axis=0, return_index=True)

        unique_fps = [fps[id] for id in idx]

        # TODO: use pdist and also select based on lowest q(x)

        return unique_fps

    @staticmethod
    def _create_fixedpoint_object(fun, fps, x0, inputs):
        """Initial creation of a fixedpoint object. A fixedpoint object is a dictionary
        providing information about a fixedpoint. Initial information is specified in
        parameters of this function. Please note that e.g. jacobian matrices will be added
        at a later stage to the fixedpoint object.

        Args:
            fun: function of dynamical system in which the fixedpoint has been optimized.

            fps: numpy array containing all optimized fixedpoints. It has the size
            [n_init x n_units]. No default.

            x0: numpy array containing all initial conditions. It has the size
            [n_init x n_units]. No default.

            inputs: numpy array containing all inputs corresponding to initial conditions.
            It has the size [n_init x n_units]. No default.

        Returns:
            List of initialized fixedpointobjects."""

        fixedpoints = []
        k = 0
        for fp in fps:

            fixedpointobject = {'fun': fun(fp),
                          'x': fp,
                          'x_init': x0[k, :],
                          'input_init': inputs[k, :]}
            fixedpoints.append(fixedpointobject)
            k += 1

        return fixedpoints

    def _compute_jacobian(self, fixedpoints):
        """Computes jacobians of fixedpoints. It is a linearization around the fixedpoint
        in all dimensions.

        Args: fixedpoints: fixedpointobject containing initialized fixed points, I.e. the object
        must a least provide fp['x'], the position of the optimized fixedpoint in its n_units
        dimensional space.

        Retruns: fixedpoints: fixedpointobject that now contains a jacobian matrix in fp['jac'].
        The jacobian matrix will have the dimensions [n_units x n_units]."""
        k = 0
        for fp in fixedpoints:
            if self.rnn_type == 'vanilla':
                fun, jac_fun = build_rnn_ds(self.weights, self.n_hidden, fp['input_init'], 'sequential')
            elif self.rnn_type == 'gru':
                fun, jac_fun = build_gru_ds(self.weights, self.n_hidden, fp['input_init'], 'sequential')
            elif self.rnn_type == 'lstm':
                fun, jac_fun = build_lstm_ds(self.weights, fp['input_init'],
                                                                   self.n_hidden, 'sequential')
            else:
                raise ValueError('Hyperparameter rnn_type must be one of'
                                 '[vanilla, gru, lstm] but was %s', self.rnn_type)

            fp['jac'] = jac_fun(fp['x'])
            k += 1

        return fixedpoints


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
              f"{wn}Architecture to analyse {ec}: {bc}{self.rnn_type}{ec}\n"
              f"The layer has {bc}{self.n_hidden}{ec} recurrent units. \n"
              f"-----------------------------------------\n"
              f"{wn}HyperParameters{ec}: \n"
              f"threshold - {self.q_threshold}\n "
              f"unique_tolerance - {self.unique_tol}\n"
              f"-----------------------------------------\n")

# TODO: make function to classify fixedpoints apart from the one inside the plotting


class Adamfixedpointfinder(FixedPointFinder):
    adam_default_hps = {'alr_hps': {'decay_rate': 0.0005},
                        'agnc_hps': {'norm_clip': 1.0,
                                     'decay_rate': 1e-03},
                        'adam_hps': {'epsilon': 1e-03,
                                     'max_iters': 5000,
                                     'method': 'joint',
                                     'print_every': 200}}

    def __init__(self, weights, rnn_type,
                 q_threshold=FixedPointFinder._default_hps['q_threshold'],
                 tol_unique=FixedPointFinder._default_hps['tol_unique'],
                 verbose=FixedPointFinder._default_hps['verbose'],
                 random_seed=FixedPointFinder._default_hps['random_seed'],
                 alr_decayr=adam_default_hps['alr_hps']['decay_rate'],
                 agnc_normclip=adam_default_hps['agnc_hps']['norm_clip'],
                 agnc_decayr=adam_default_hps['agnc_hps']['decay_rate'],
                 epsilon=adam_default_hps['adam_hps']['epsilon'],
                 max_iters = adam_default_hps['adam_hps']['max_iters'],
                 method = adam_default_hps['adam_hps']['method'],
                 print_every = adam_default_hps['adam_hps']['print_every']):

        FixedPointFinder.__init__(self, weights, rnn_type, q_threshold=q_threshold,
                                  tol_unique=tol_unique,
                                  verbose=verbose,
                                  random_seed=random_seed)
        self.alr_decayr = alr_decayr
        self.agnc_normclip = agnc_normclip
        self.agnc_decayr = agnc_decayr
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.method = method
        self.print_every = print_every

        if self.verbose:
            self._print_adam_hps()

    def _print_adam_hps(self):
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
        print(f"{wn}HyperParameters for adam{ec}: \n"
              f" learning rate - {self.epsilon}\n "
              f"maximum iterations - {self.method}\n"
              f"print every {self.print_every} iterations\n"
              f"performing {self.method} optimization\n"
              f"-----------------------------------------\n")

    def find_fixed_points(self, x0, inputs):
        """Compute fixedpoints and determine the uniqueness as well as jacobian matrix.
            Optimization can occur either joint or sequential.

        Args:
            x0: set of initial conditions to optimize from.

            inputs: set of inputs to recurrent layer corresponding to initial conditions.

        Returns:
            List of fixedpointobjects that are unique and equipped with their repsective
            jacobian matrix."""

        if self.method == 'joint':
            fixedpoints = self._joint_optimization(x0, inputs)
        elif self.method == 'sequential':
            fixedpoints = self._sequential_optimization(x0, inputs)
        else:
            raise ValueError('HyperParameter for adam optimizer: method must be either "joint"'
                             'or "sequential". However, it was %s', self.method)

        good_fps, bad_fps = self._handle_bad_approximations(fixedpoints)
        unique_fps = self._find_unique_fixed_points(good_fps)

        unique_fps = self._compute_jacobian(unique_fps)

        return unique_fps

    def _joint_optimization(self, x0, inputs):
        """Function to perform joint optimization. All inputs and initial conditions
        will be optimized simultaneously.

        Args:
            x0: numpy array containing a set of initial conditions. Must have dimensions
            [n_init x n_units]. No default.

            inputs: numpy array containing a set of inputs to the recurrent layer corresponding
            to the activations in x0. Must have dimensions [n_init x n_units]. No default.

        Returns:
            Fixedpointobject. See _create_fixedpoint_object for further documenation."""

        if self.rnn_type == 'vanilla':
            fun, jac_fun = build_rnn_ds(self.weights, self.n_hidden, inputs, self.method)
        elif self.rnn_type == 'gru':
            fun, jac_fun = build_gru_ds(self.weights, self.n_hidden, inputs, self.method)
        elif self.rnn_type == 'lstm':
            fun, jac_fun = build_lstm_ds(self.weights, inputs, self.n_hidden, self.method)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.rnn_type)

        if self.rnn_type == 'lstm':
            fps = adam_lstm(fun, x0,
                            epsilon=self.epsilon,
                            alr_decayr=self.alr_decayr,
                            max_iter=self.max_iters,
                            print_every=self.print_every,
                            init_agnc=self.agnc_normclip,
                            agnc_decayr=self.agnc_decayr,
                            verbose=self.verbose)
            fun, jac_fun = build_lstm_ds(self.weights, inputs,
                                                               self.n_hidden, 'sequential')
            fixedpoints = self._create_fixedpoint_object(fun, fps, x0, inputs)
        else:
            fps = adam_optimizer(fun, x0,
                                 epsilon=self.epsilon,
                                 alr_decayr=self.alr_decayr,
                                 max_iter=self.max_iters,
                                 print_every=self.print_every,
                                 init_agnc=self.agnc_normclip,
                                 agnc_decayr=self.agnc_decayr,
                                 verbose=self.verbose)

            fixedpoints = self._create_fixedpoint_object(fun, fps, x0, inputs)

        return fixedpoints

    def _sequential_optimization(self, x0, inputs):
        """Function to perform sequential optimization. All inputs and initial conditions
        will be optimized individually.

        Args:
            x0: numpy array containing a set of initial conditions. Must have dimensions
            [n_init x n_units]. No default.

            inputs: numpy array containing a set of inputs to the recurrent layer corresponding
            to the activations in x0. Must have dimensions [n_init x n_units]. No default.

        Returns:
            Fixedpointobject. See _create_fixedpoint_object for further documenation."""
        fps = np.empty(x0.shape)
        for i in range(len(x0)):
            if self.rnn_type == 'vanilla':
                fun, jac_fun = build_rnn_ds(self.weights, self.n_hidden, inputs[i, :], self.method)
            elif self.rnn_type == 'gru':
                fun, jac_fun = build_gru_ds(self.weights, self.n_hidden, inputs[i, :], self.method)
            elif self.rnn_type == 'lstm':
                fun, jac_fun = build_lstm_ds(self.weights, inputs[i, :], self.n_hidden, self.method)
            else:
                raise ValueError('Hyperparameter rnn_type must be one of'
                                 '[vanilla, gru, lstm] but was %s', self.rnn_type)
        # TODO: implement parallel sequential optimization

            fps[i, :] = adam_optimizer(fun, x0[i, :],
                                       epsilon=self.epsilon,
                                       alr_decayr=self.alr_decayr,
                                       max_iter=self.max_iters,
                                       print_every=self.print_every,
                                       init_agnc=self.agnc_normclip,
                                       agnc_decayr=self.agnc_decayr,
                                       verbose=self.verbose)

        fixedpoints = self._create_fixedpoint_object(fun, fps, x0, inputs)

        return fixedpoints


class Scipyfixedpointfinder(FixedPointFinder):
    scipy_default_hps = {'method': 'Newton-CG',
                         'xtol': 1e-20,
                         'display': True}

    def __init__(self, weights, rnn_type,
                 q_threshold=FixedPointFinder._default_hps['q_threshold'],
                 tol_unique=FixedPointFinder._default_hps['tol_unique'],
                 verbose=FixedPointFinder._default_hps['verbose'],
                 random_seed=FixedPointFinder._default_hps['random_seed'],
                 method=scipy_default_hps['method'],
                 xtol=scipy_default_hps['xtol'],
                 display=scipy_default_hps['display']):
        FixedPointFinder.__init__(self, weights, rnn_type,
                                  q_threshold=q_threshold,
                                  tol_unique=tol_unique,
                                  verbose=verbose,
                                  random_seed=random_seed)
        self.method = method
        self.xtol = xtol
        self.display = display

    def find_fixed_points(self, x0, inputs):
        """"""
        pool = mp.Pool(mp.cpu_count())
        combined_objects = []
        for i in range(len(x0)):
            combined_objects.append((x0[i, :], inputs[i, :], self.rnn_type,
                                     self.n_hidden, self.weights, self.method,
                                     self.xtol, self.display))
        fps = pool.map(self._scipy_optimization, combined_objects, chunksize=1)
        fps = [item for sublist in fps for item in sublist]

        good_fps, bad_fps = self._handle_bad_approximations(fps)
        unique_fps = self._find_unique_fixed_points(good_fps)

        unique_fps = self._compute_jacobian(unique_fps)

        return unique_fps

    @staticmethod
    def _scipy_optimization(combined_object):
        """Optimization using minimize from scipy. Theoretically all availabel algorithms
        are usable while some are not suitable for the problem. Thus, the hyperparameter method
        should be set to method that make use of the hessian matrix e.g. 'Newton-CG' or 'trust-ncg'.

        Args:
            combined_object: Single tuple containing eight objects listed below. Due to parallelization
            as single object was required

            x0: Vector with Initial Condition to begin the minimization from. Must have dimensions
            [n_units]. No default.

            inputs: Vector with input corresponding to Initial Condition. Must have dimensions [n_units].
            No default.

            rnn_type: String specifying the rnn_type. No default.

            n_hidden: Integer indicating number of hidden units. No default.

            weights: List of matrices containing recurrent weights, input weights and biases.
            Structure depends on architecture to analyze. No default.

            method: Method to pass to minimize to use for optimization. Default: 'Newton-CG'.

            xtol: tolerance to pass to minimize. Interpretation of tolerance depends on chosen
            method. See scipy documentation for further information.

            display: boolean indicating whether minimize should print conversion messages.
            Default: True.

        Returns:
            Fixedpoint dictionary containing a single fixed point."""
        x0, inputs, rnn_type, n_hidden, weights, method, xtol, display = combined_object[0], combined_object[1], \
                                                                         combined_object[2], combined_object[3], \
                                                                         combined_object[4], combined_object[5], \
                                                                         combined_object[6], combined_object[7]

        if rnn_type == 'vanilla':
            fun, jac_fun = build_rnn_ds(weights, n_hidden, inputs, 'sequential')
        elif rnn_type == 'gru':
            fun, jac_fun = build_gru_ds(weights, n_hidden, inputs, 'sequential')
        elif rnn_type == 'lstm':
            fun, jac_fun = build_lstm_ds(weights, inputs, n_hidden, 'sequential')
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', rnn_type)


        jac, hes = nd.Gradient(fun), nd.Hessian(fun)
        options = {'disp': display}

        fp = minimize(fun, x0, method=method, jac=jac, hess=hes,
                      options=options, tol=xtol)

        fpx = fp.x.reshape((1, len(fp.x)))
        x0 = x0.reshape((1, len(x0)))
        inputs = inputs.reshape((1, len(inputs)))

        fixedpoint = FixedPointFinder._create_fixedpoint_object(fun, fpx, x0, inputs)
        return fixedpoint


class Tffixedpointfinder(FixedPointFinder):
    tf_default_hps = {'alr_hps': {'decay_rate': 0.0005},
                        'agnc_hps': {'norm_clip': 1.0,
                                     'decay_rate': 1e-03},
                        'adam_hps': {'epsilon': 1e-03,
                                     'max_iters': 5000,
                                     'method': 'joint',
                                     'print_every': 200}}

    def __init__(self, weights, rnn_type,
                 q_threshold=FixedPointFinder._default_hps['q_threshold'],
                 tol_unique=FixedPointFinder._default_hps['tol_unique'],
                 verbose=FixedPointFinder._default_hps['verbose'],
                 random_seed=FixedPointFinder._default_hps['random_seed'],
                 alr_decayr=tf_default_hps['alr_hps']['decay_rate'],
                 agnc_normclip=tf_default_hps['agnc_hps']['norm_clip'],
                 agnc_decayr=tf_default_hps['agnc_hps']['decay_rate'],
                 epsilon=tf_default_hps['adam_hps']['epsilon'],
                 max_iters=tf_default_hps['adam_hps']['max_iters'],
                 method=tf_default_hps['adam_hps']['method'],
                 print_every=tf_default_hps['adam_hps']['print_every']):
        FixedPointFinder.__init__(self, weights, rnn_type, q_threshold=q_threshold,
                                  tol_unique=tol_unique,
                                  verbose=verbose,
                                  random_seed=random_seed)
        self.alr_decayr = alr_decayr
        self.agnc_normclip = agnc_normclip
        self.agnc_decayr = agnc_decayr
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.method = method
        self.print_every = print_every

        if self.verbose:
            self._print_tf_hps()

    def _print_tf_hps(self):
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
        print(f"{wn}HyperParameters for adam{ec}: \n"
              f" learning rate - {self.epsilon}\n "
              f"maximum iterations - {self.method}\n"
              f"print every {self.print_every} iterations\n"
              f"performing {self.method} optimization\n"
              f"-----------------------------------------\n")

    def find_fixed_points(self, states, inputs, model):

        lstmcell = build_sub_model_to(model, [self.rnn_type])
        # init_c_h = states
        # input = np.vstack(stim['inputs'][:20, :, :])
        # input = np.zeros((n, 3))
        # c = init_c_h[0:n, n_hidden:]  # [n_batch x n_dims]
        # h = init_c_h[0:n, :n_hidden]  # [n_batch x n_dims]

        inputs = tf.constant(inputs, dtype='float32')

        # tuple = tf.Variable([tf.convert_to_tensor(h), tf.convert_to_tensor(c)],
        #   dtype='float32')

        # output, next_state = lstmcell(inputs, tuple)
        # lstmcell.set_weights(weights)
        # lstmcell.trainable = False
        # tuple = tf.reshape(tuple, [2, 2*n_hidden])
        # F = tf.concat((next_state[0], next_state[1]), axis=1)

        # optimizer = tf.keras.optimizers.Adam(0.001)
        # for i in range(5000):
        #    with tf.GradientTape() as tape:
        #       q = 0.5 * tf.reduce_sum(tf.square(next_state - tuple), axis=1)
        #       q_scalar = tf.reduce_mean(q)
        #       gradients = tape.gradient(q_scalar, [tuple])
        #   optimizer.apply_gradients(zip(gradients, [tuple]))

        #  print(q_scalar.numpy())
        #  print(tf.reduce_mean(tuple))
        # jac = tape.jacobian(q, [tuple])

