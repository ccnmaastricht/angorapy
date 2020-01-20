from scipy.optimize import minimize
import numpy as np
import numdifftools as nd
import multiprocessing as mp


def minimizGRU(self, weights, inputweights, inputs, activation, method: str = "trust-ncg"):
    pass


def minimizRNN(combined):
    x0, input, weights, inputweights, method = combined[0][0:64], combined[0][64:128], combined[1], combined[2], \
                                               combined[3]
    fun = lambda x: 0.5 * sum(
        (- x[0:64] + np.matmul(weights, np.tanh(x[0:64])) + np.matmul(inputweights, input)) ** 2)
    options = {'gtol': 1e-10, 'disp': True}
    jac, hes = nd.Gradient(fun), nd.Hessian(fun)
    y = fun(x0)
    print("First function evaluation:", y)
    optimised_result = minimize(fun, x0, method=method, jac=jac, hess=hes,
                                options=options)
    return optimised_result


def parallel_minimization(inputs, activation, weights, inputweights, method):
    """Function to set up parrallel processing of minimization for fixed points


    """
    pool = mp.Pool(mp.cpu_count())
    ids = np.arange(0, activation.shape[0], 10)
    print(len(ids), " minimizations to parallelize.")
    x0, input = activation[ids, :], inputs[ids, :]
    combind = []
    for i in range(x0.shape[0]):
        combind.append((np.hstack((x0[i, :], input[i, :])), weights, inputweights, method))

    optimised_results = pool.map(minimizRNN, combind, chunksize=1)

    return optimised_results
