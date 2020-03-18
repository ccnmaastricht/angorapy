import autograd.numpy as np
def build_rnn_inducer(target):

    def fun(x):
        return np.mean(0.5 * np.sum(((- target + np.matmul(np.tanh(target), x)) ** 2), axis=1))

    return fun


def build_gru_inducer(target, n_hidden):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    z_fun = lambda x: sigmoid(np.matmul(target, x))
    r_fun = lambda x: sigmoid(np.matmul(target, x))
    def g_fun(x):
        r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden)
        return np.tanh((r_fun(x[:, r]) * np.matmul(target, x[:, h])))

    def fun(x):
        z, rh = np.arange(0, n_hidden), np.arange(n_hidden, 3 * n_hidden)
        return np.mean(0.5 * np.sum((((1 - z_fun(x[:, z])) * (g_fun(x[:, rh]) - target)) ** 2), axis=1))

    return fun