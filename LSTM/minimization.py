from scipy.optimize import minimize
import numpy as np
import numdifftools as nd
import multiprocessing as mp
from functools import partial

class Minimizer:

    def __init__(self, weights, inputweights, method: str = "trust-ncg"):
        self.weights = weights
        self.inputweights = inputweights
        self.method = method

    def minimizGRU(self, weights, inputweights, inputs, activation, method: str = "trust-ncg"):
        pass

def minimizRNN(weights, inputweights, input, activation, method):
    # optimisedResults = []
    # id = np.random.randint(activation.shape[0])
    # ids = np.arange(0, activation.shape[0], 10)
    # print("Timestep:", ids[id])
    x0 = activation
    # input = inputs[ids[id], :]
    fun = lambda x: 0.5 * sum(
        (- x[0:64] + np.matmul(weights, np.tanh(x[0:64])) + np.matmul(inputweights, input)) ** 2)
    options = {'gtol': 1e-10, 'disp': True}
    Jac, Hes = nd.Gradient(fun), nd.Hessian(fun)
    y = fun(x0)
    print("First function evaluation:", y)
    optimisedResult = minimize(fun, x0, method=method, jac=Jac, hess=Hes,
                               options=options)
    # optimisedResults.append(optimisedResult)
    return optimisedResult
def wrapper(combined):
    x0, input, weights, inputweights, method = combined[0][0:64], combined[0][64:128], combined[1], combined[2], combined[3]
    optimisedResults = minimizRNN(weights, inputweights, input=input, activation=x0, method=method)
    return optimisedResults

def parallelised(inputs, activation, weights, inputweights, method):
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)
    ids = np.arange(0, activation.shape[0], 10)
    x0 = activation[ids, :]
    input = inputs[ids, :]
    combind = []
    for i in range(x0.shape[0]):
        combind.append((np.hstack((x0[i, :], input[i, :])), weights, inputweights, method))



    optimisedResults = pool.map(wrapper, combind, chunksize=1)

    return optimisedResults