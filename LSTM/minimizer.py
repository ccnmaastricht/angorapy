# this file contains the great minimizer
from scipy.optimize import minimize
import numpy as np
import numdifftools as nd

class Minimizer:

    def __init__(self):
        pass

    def minimizGRU(self, weights, inputweights, inputs, activation, method: str = "trust-ncg"):

    def minimizRNN(self, weights, inputweights, inputs, activation, method: str = 'trust-ncg'):
        optimisedResults = []
        # id = np.random.randint(activation.shape[0])
        ids = np.arange(0, activation.shape[0], 10)
        for id in range(len(ids)):
            print("Timestep:", ids[id])
            x0 = activation[ids[id], :]
            input = inputs[ids[id], :]
            fun = lambda x: 0.5 * sum(
                (- x[0:64] + np.matmul(weights, np.tanh(x[0:64])) + np.matmul(inputweights, input)) ** 2)
            options = {'gtol': 1e-5, 'disp': True}
            Jac, Hes = nd.Gradient(fun), nd.Hessian(fun)
            y = fun(x0)
            print("First function evaluation:", y)
            optimisedResult = minimize(fun, x0, method=method, jac=Jac, hess=Hes,
                                       options=options)
            optimisedResults.append(optimisedResult)
        return optimisedResults