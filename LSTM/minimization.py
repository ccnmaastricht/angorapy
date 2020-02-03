import numpy as np
import numdifftools as nd



def backproprnn(combined):
    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))
    x0, input, weights, n_hidden = combined[0], combined[1], combined[2], combined[3]
    n_hidden = 24
    weights, inputweights, b = weights[1], weights[0], weights[2]
    projection_b = np.matmul(input, inputweights) + b
    fun = lambda x: 0.5 * sum((- x[0:n_hidden] + np.matmul(np.tanh(x[0:n_hidden]), weights) + projection_b) ** 2)
    grad_fun = nd.Gradient(fun)
    lr = 0.1
    for i in range(500):
        q = fun(x0)
        print(q)
        dq = grad_fun(x0)
        dq = np.round(dq, 15)
        x0 = x0 - lr * dq
        lr = decay_lr(0.1, 0.001, i)
    print('new IC')
    fixedpoint = x0

    return fixedpoint

def backpropgru(combined):
    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

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
    lr = 0.1
    for i in range(1000):
        q = fun(x0)
        print(q)
        dq = grad_fun(x0)
        dq = np.round(dq, 15)
        x0 = x0 - lr * dq
        lr = decay_lr(0.1, 0.001, i)
    print('new IC')
    fixedpoint = x0

    return fixedpoint



