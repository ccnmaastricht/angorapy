import numpy as np
import multiprocessing as mp
import numdifftools as nd
from scipy.optimize import minimize

class FixedPointFinder:
    def __init__(self, hps, data_hps):
        self.hps = hps
        self.data_hps = data_hps


    def parallel_minimization(self, weights, inputweights, activation, inputs, method):
        """Function to set up parrallel processing of minimization for fixed points


        """
        pool = mp.Pool(mp.cpu_count())
        ids = np.arange(0, activation.shape[0])
        print(len(ids), " minimizations to parallelize.")
        x0, input = activation[ids, :], inputs[ids, :]
        combind = []
        for i in range(x0.shape[0]):
            combind.append((x0[i, :], input[i, :], weights, inputweights, method))

        self.fixedpoints = pool.map(self._minimizRNN, combind, chunksize=1)
        return self.fixedpoints


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

# TODO: implement other architectures
# TODO: need handling of bad approximations
