import autograd.numpy as np
import numdifftools as nd


def build_rnn_ds(weights, n_hidden, inputs, method: str = 'joint'):
    weights, inputweights, b = weights[1], weights[0], weights[2]
    projection_b = np.matmul(inputs, inputweights) + b

    if method == 'joint':
        def fun(x):
            return np.mean(0.5 * np.sum(((- x + np.matmul(np.tanh(x), weights) + projection_b) ** 2), axis=1))
    elif method == 'sequential':
        def fun(x):
            return 0.5 * np.sum((- x + np.matmul(np.tanh(x), weights) + projection_b) ** 2)
    elif method == 'velocity':
        def fun(x):
            return 0.5 * np.sum(((- x + np.matmul(np.tanh(x), weights) + projection_b) ** 2), axis=1)
    else:
        raise ValueError('Method argument to build function must be one of '
                         '[joint, sequential, velocity] but was', method)

    jac_fun = lambda x: - np.eye(n_hidden, n_hidden) + weights * (1 - np.tanh(x) ** 2)

    return fun, jac_fun


def build_gru_ds(weights, n_hidden, input, method: str = 'joint'):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden), np.arange(2 * n_hidden, 3 * n_hidden)
    W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
    U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
    b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

    z_projection_b = np.matmul(input, W_z) + b_z
    r_projection_b = np.matmul(input, W_r) + b_r
    g_projection_b = np.matmul(input, W_h) + b_h

    z_fun = lambda x: sigmoid(np.matmul(x, U_z) + z_projection_b)
    r_fun = lambda x: sigmoid(np.matmul(x, U_r) + r_projection_b)
    g_fun = lambda x: np.tanh((r_fun(x) * np.matmul(x, U_h) + g_projection_b))

    if method == 'joint':
        def fun(x):
            return np.mean(0.5 * np.sum((((1 - z_fun(x)) * (g_fun(x) - x)) ** 2), axis=1))
    elif method == 'sequential':
        fun = lambda x: 0.5 * np.sum(((1 - z_fun(x)) * (g_fun(x) - x)) ** 2)
    elif method == 'velocity':
        fun = lambda x: 0.5 * np.sum((((1 - z_fun(x)) * (g_fun(x) - x)) ** 2), axis=1)
    else:
        raise ValueError('Method argument to build function must be one of '
                     '[joint, sequential, velocity] but was', method)

    def dynamical_system(x):
        return (1 - z_fun(x)) * (g_fun(x) - x)
    jac_fun = nd.Jacobian(dynamical_system)

    return fun, jac_fun


def build_lstm_ds(weights, input, n_hidden, method: str = 'joint'):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    W, U, b = weights[0], weights[1], weights[2]

    W_i, W_f, W_c, W_o = W[:, :n_hidden], W[:, n_hidden:2 * n_hidden], \
                         W[:, 2 * n_hidden:3 * n_hidden], W[:, 3 * n_hidden:]
    U_i, U_f, U_c, U_o = U[:, :n_hidden], U[:, n_hidden:2 * n_hidden], \
                         U[:, 2 * n_hidden:3 * n_hidden], U[:, 3 * n_hidden:]
    b_i, b_f, b_c, b_o = b[:n_hidden], b[n_hidden:2 * n_hidden], \
                         b[2 * n_hidden:3 * n_hidden], b[3 * n_hidden:]

    f_projection_b = np.matmul(input, W_f) + b_f
    i_projection_b = np.matmul(input, W_i) + b_i
    o_projection_b = np.matmul(input, W_o) + b_o
    c_projection_b = np.matmul(input, W_c) + b_c

    f_fun = lambda x: sigmoid(np.matmul(x, U_f) + f_projection_b)
    i_fun = lambda x: sigmoid(np.matmul(x, U_i) + i_projection_b)
    o_fun = lambda x: sigmoid(np.matmul(x, U_o) + o_projection_b)
    c_fun = lambda c, h: f_fun(h) * c + i_fun(h) * np.tanh((np.matmul(h, U_c) + c_projection_b))

    if method == 'joint':
        def h_fun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return c_fun(c, h)

        def fun(x):
            return np.mean(0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x), axis=1))

    elif method == 'sequential':
        def h_fun(x):
            c, h = x[n_hidden:], x[:n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[n_hidden:], x[:n_hidden]
            return c_fun(c, h)

        def fun(x):
            return 0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x))

    elif method == 'velocity':
        def h_fun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return c_fun(c, h)

        def fun(x):
            return 0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x), axis=1)
    else:
        raise ValueError('Method argument to build function must be one of '
                         '[joint, sequential, velocity] but was', method)

    def dynamical_system(x):
        return np.hstack((h_fun(x), cfun(x))) - x

    jac_fun = nd.Jacobian(dynamical_system)

    return fun, jac_fun


