import numpy as np
import numdifftools as nd



def backproprnn(combined):

    def print_update(q, lr):
        print("Function value:", q,"; lr:", np.round(lr, 5), "; ")


    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))
    x0, input, weights, n_hidden = combined[0], combined[1], combined[2], combined[3]
    n_hidden = 24
    weights, inputweights, b = weights[1], weights[0], weights[2]
    projection_b = np.matmul(input, inputweights) + b
    fun = lambda x: 0.5 * np.sum((- x[0:n_hidden] + np.matmul(np.tanh(x[0:n_hidden]), weights)) **2 )
    grad_fun = nd.Gradient(fun)
    inlr = 0.001
    for i in range(500):
        q = fun(x0)
        dq = grad_fun(x0)
        dq = np.round(dq, 15)
        lr = decay_lr(inlr, 0.001, i)
        x0 = x0 - lr * dq

        if i % 100 == 0:
            print_update(q, lr)
    print('new IC')
    jac_fun = lambda x: - np.eye(n_hidden, n_hidden) + weights * (1 - np.tanh(x[0:n_hidden]) ** 2)
    jacobian = jac_fun(x0)
    fixedpoint = {'fun': q,
                  'x': x0,
                  'jac': jacobian}

    return fixedpoint

def backpropgru(combined):
    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def print_update(q, lr):
        print("Function value:", q, "; lr:", np.round(lr, 5), "; ")


    x0, input, weights, n_hidden = combined[0], \
                                           combined[1], \
                                           combined[2], combined[3]
    z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden), np.arange(2 * n_hidden, 3 * n_hidden)
    W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
    U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
    b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

    z_projection_b = np.matmul(input, W_z) + b_z
    r_projection_b = np.matmul(input, W_r) + b_r
    g_projection_b = np.matmul(input, W_h) + b_h

    z_fun = lambda x: sigmoid(np.matmul(x[0:n_hidden], U_z) + b_z)
    r_fun = lambda x: sigmoid(np.matmul(x[0:n_hidden], U_r) + b_r)
    g_fun = lambda x: np.tanh((r_fun(x[0:n_hidden]) * np.matmul(x[0:n_hidden], U_h) + b_h))

    fun = lambda x: 0.5 * sum(((1 - z_fun(x[0:n_hidden])) * (g_fun(x[0:n_hidden]) - x[0:n_hidden])) ** 2)
    grad_fun = nd.Gradient(fun)
    inlr = 0.01
    max_iter = 4000
    maxclip, minclip = 30, 10
    gradientclip = np.repeat(np.arange(minclip, maxclip, 1), int(max_iter/(maxclip-minclip)))

    for i in range(max_iter):
        q = fun(x0)
        dq = grad_fun(x0)
        dq = np.round(dq, gradientclip[i])
        lr = decay_lr(inlr, 0.0001, i)
        x0 = x0 - lr * dq
        if i % 100 == 0:
            print_update(q, lr)
    dynamical_system = lambda x: (1 - z_fun(x[0:n_hidden])) * (g_fun(x[0:n_hidden]) - x[0:n_hidden])
    jac_fun = nd.Jacobian(dynamical_system)
    jacobian = jac_fun(x0)
    print('Function value after minimization ', q)
    fixedpoint = {'fun': q,
                  'x': x0,
                  'jac': jacobian}

    return fixedpoint

def exponential_minimization():
    linspace = np.linspace(minima, maxima, num=100)



