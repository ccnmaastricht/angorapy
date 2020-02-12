import autograd.numpy as np
import numdifftools as nd


def build_rnn_ds(weights, n_hidden, inputs, method: str = 'joint'):
    weights, inputweights, b = weights[1], weights[0], weights[2]
    projection_b = np.matmul(inputs, inputweights) + b

    if method == 'joint':
        def fun(x):
            return np.mean(0.5 * sum((- x + np.matmul(np.tanh(x), weights) + projection_b) ** 2))
    else:
        def fun(x):
            return 0.5 * sum((- x + np.matmul(np.tanh(x), weights) + projection_b) ** 2)

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
    else:
        def fun(x):
            return 0.5 * np.sum((((1 - z_fun(x)) * (g_fun(x) - x)) ** 2), axis=1)

    def dynamical_system(x):
        return (1 - z_fun(x)) * (g_fun(x) - x)
    jac_fun = nd.Jacobian(dynamical_system)

    return fun, jac_fun

def build_lstm_ds(weights, input, n_hidden, method: str = 'joint'):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    W, U, b = weights[0], weights[1], weights[2]

    W_i, W_f, W_c, W_o = W[:, :n_hidden], W[:, n_hidden:2 * n_hidden], \
                         W[:, 2 * n_hidden:3 * n_hidden], W[:, 3 * n_hidden:4 * n_hidden]
    U_i, U_f, U_c, U_o = U[:, :n_hidden].transpose(), U[:, n_hidden:2 * n_hidden].transpose(), \
                         U[:, 2 * n_hidden:3 * n_hidden].transpose(), U[:, 3 * n_hidden:4 * n_hidden].transpose()
    b_i, b_f, b_c, b_o = b[0, :n_hidden].transpose(), b[0, n_hidden:2 * n_hidden].transpose(), \
                         b[0, 2 * n_hidden:3 * n_hidden].transpose(), b[0, 3 * n_hidden:4 * n_hidden].transpose()
    f_fun = lambda x: sigmoid(np.matmul(input, W_f) + np.matmul(U_f, x[0:n_hidden]) + b_f)
    i_fun = lambda x: sigmoid(np.matmul(input, W_i) + np.matmul(U_i, x[0:n_hidden]) + b_i)
    o_fun = lambda x: sigmoid(np.matmul(input, W_o) + np.matmul(U_o, x[0:n_hidden]) + b_o)
    c_fun = lambda x, c: f_fun(x[0:n_hidden]) * c[0:n_hidden] + i_fun(x[0:n_hidden]) * \
                         np.tanh((np.matmul(W_c, input) + np.matmul(U_c, x[0:n_hidden]) + b_c) - c[0:n_hidden])
    # perhaps put h and c in as one object to minimize and split up in functions
    if method == 'joint':
        fun = lambda x, c: np.mean(0.5 * sum((o_fun(x[0:n_hidden]) * np.tanh(c_fun(x[0:n_hidden], c)) - x[0:n_hidden]) ** 2))
    else:
        fun = lambda x, c: 0.5 * sum((o_fun(x[0:n_hidden]) * np.tanh(c_fun(x[0:n_hidden], c)) - x[0:n_hidden]) ** 2)

    dynamical_system = lambda x, c: o_fun(x[0:n_hidden]) * np.tanh(c_fun(x[0:n_hidden], c)) - x[0:n_hidden]
    jac_fun = nd.Jacobian(dynamical_system)

    return fun, jac_fun


