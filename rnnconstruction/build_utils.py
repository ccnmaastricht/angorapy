import numpy as np
def build_rnn_ds(target):

    def fun(x):
        return np.mean(0.5 * np.sum(((- target + np.matmul(np.tanh(target), x)) ** 2), axis=1))

    return fun

def build_gru_ds(target):

    def fun(x):
        pass

    return fun