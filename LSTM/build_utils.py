import numpy as np


def build_rnn_ds(weights, input: None, use_input: bool = False):
    weights, inputweights, b = weights[1], weights[0], weights[2]
    projection_b = np.matmul(input, inputweights) + b

    if use_input:
        fun = lambda x: 0.5 * sum((- x + np.matmul(np.tanh(x), weights) + projection_b) ** 2)
    else:
        fun = lambda x: 0.5 * sum((- x + np.matmul(np.tanh(x), weights) + b) ** 2)

    return fun


def build_gru_ds(weights, n_hidden, input: None, use_input: bool = False):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden), np.arange(2 * n_hidden, 3 * n_hidden)
    W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
    U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
    b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

    z_projection_b = np.matmul(input, W_z) + b_z
    r_projection_b = np.matmul(input, W_r) + b_r
    g_projection_b = np.matmul(input, W_h) + b_h
    if use_input:
        z_fun = lambda x: sigmoid(np.matmul(x, U_z) + z_projection_b)
        r_fun = lambda x: sigmoid(np.matmul(x, U_r) + r_projection_b)
        g_fun = lambda x: np.tanh((r_fun(x) * np.matmul(x, U_h) + g_projection_b))

        fun = lambda x: 0.5 * sum(((1 - z_fun(x)) * (g_fun(x) - x)) ** 2)
    else:
        z_fun = lambda x: sigmoid(np.matmul(x, U_z) + b_z)
        r_fun = lambda x: sigmoid(np.matmul(x, U_r) + b_r)
        g_fun = lambda x: np.tanh((r_fun(x) * np.matmul(x, U_h) + b_h))

        fun = lambda x: 0.5 * np.sum((((1 - z_fun(x)) * (g_fun(x) - x)) ** 2), axis=1)

    dynamical_system = lambda x: (1 - z_fun(x)) * (g_fun(x) - x)

    return fun, dynamical_system

def build_lstm_ds(weights, input, n_hidden, use_input: bool = False):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    W, U, b = weights[0], weights[1], weights[2]

    W_i, W_f, W_c, W_o = W[:, :n_hidden].transpose(), W[:, n_hidden:2 * n_hidden].transpose(), \
                         W[:, 2 * n_hidden:3 * n_hidden].transpose(), W[:, 3 * n_hidden:4 * n_hidden].transpose()
    U_i, U_f, U_c, U_o = U[:, :n_hidden].transpose(), U[:, n_hidden:2 * n_hidden].transpose(), \
                         U[:, 2 * n_hidden:3 * n_hidden].transpose(), U[:, 3 * n_hidden:4 * n_hidden].transpose()
    b_i, b_f, b_c, b_o = b[0, :n_hidden].transpose(), b[0, n_hidden:2 * n_hidden].transpose(), \
                         b[0, 2 * n_hidden:3 * n_hidden].transpose(), b[0, 3 * n_hidden:4 * n_hidden].transpose()
    f_fun = lambda x: sigmoid(np.matmul(W_f, input) + np.matmul(U_f, x[0:n_hidden]) + b_f)
    i_fun = lambda x: sigmoid(np.matmul(W_i, input) + np.matmul(U_i, x[0:n_hidden]) + b_i)
    o_fun = lambda x: sigmoid(np.matmul(W_o, input) + np.matmul(U_o, x[0:n_hidden]) + b_o)
    c_fun = lambda x, c: f_fun(x[0:n_hidden]) * c[0:n_hidden] + i_fun(x[0:n_hidden]) * \
                         np.tanh((np.matmul(W_c, input) + np.matmul(U_c, x[0:n_hidden]) + b_c) - c[0:n_hidden])
    # perhaps put h and c in as one object to minimize and split up in functions
    fun = lambda x, c: 0.5 * sum((o_fun(x[0:n_hidden]) * np.tanh(c_fun(x[0:n_hidden], c)) - x[0:n_hidden]) ** 2)

    dynamical_system = lambda x, c: o_fun(x[0:n_hidden]) * np.tanh(c_fun(x[0:n_hidden], c)) - x[0:n_hidden]

    return fun, dynamical_system
