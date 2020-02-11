import autograd.numpy as np
import numdifftools as nd
from autograd import grad
from LSTM.build_utils import build_rnn_ds, build_gru_ds



def backproprnn(combined):

    def print_update(q):
        print("Function value:", q)

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    x0, input, weights, n_hidden, adam_hps, use_input = combined[0], combined[1], combined[2], combined[3], \
                                                        combined[4], combined[5]

    fun = build_rnn_ds(weights, input, use_input=use_input)
    # grad_fun = nd.Gradient(fun)
    inlr, max_iter = adam_hps['lr'], adam_hps['max_iter']
    beta_1, beta_2 = 0.9, 0.999
    m, v = np.zeros(n_hidden), np.zeros(n_hidden)
    epsilon = 1e-08
    for t in range(max_iter):
        q = fun(x0)
        lr = decay_lr(inlr, 0.0001, t)
        # gradient norm clip will be implemented here.
        dq = grad(fun)(x0)
        norm = np.linalg.norm(dq)
        if norm > adam_hps['gradientnormclip']:
            dq = dq/norm
        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t+1))
        v_hat = v / (1 - np.power(beta_2, t+1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        if t % adam_hps['print_every'] == 0:
            print_update(q)

    fixedpoints = []
    jac_fun = lambda x: - np.eye(n_hidden, n_hidden) + weights * (1 - np.tanh(x) ** 2)
    for i in range(adam_hps['n_init']):
        jacobian = jac_fun(x0[i, :])
        fixedpoint = {'fun': q,
                      'x': x0[i, :],
                      'jac': jacobian}
        fixedpoints.append(fixedpoint)

    return fixedpoints

def backpropgru(combined):

    def print_update(q):
        print("Function value:", q, ";")

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))


    x0, input, weights, n_hidden, adam_hps = combined[0], \
                                           combined[1], \
                                           combined[2], combined[3], combined[4],
    fun, dynamical_system = build_gru_ds(weights, n_hidden, input, use_input=False)
    # grad_fun = nd.Gradient(fun)
    inlr, max_iter = adam_hps['lr'], adam_hps['max_iter']
    beta_1, beta_2 = 0.9, 0.999
    m, v = np.zeros(n_hidden), np.zeros(n_hidden)
    epsilon = 1e-08
    for t in range(max_iter):
        q = fun(x0)
        # dq = grad_fun(x0)
        lr = decay_lr(inlr, 0.0001, t)
        dq = grad(fun)(x0)
        norm = np.linalg.norm(dq)
        if norm > adam_hps['gradientnormclip']:
            dq = dq/norm

        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t+1))
        v_hat = v / (1 - np.power(beta_2, t+1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        if t % adam_hps['print_every'] == 0:
            print_update(q)

    jac_fun = nd.Jacobian(dynamical_system)
    fixedpoints = []
    for i in range(adam_hps['n_init']):
        jacobian = jac_fun(x0[i, :])
        fixedpoint = {'fun': q,
                      'x': x0[i, :],
                      'jac': jacobian}
        fixedpoints.append(fixedpoint)

    return fixedpoints


def adam_optimizer(fun, x0,
                   epsilon,
                   max_iter,
                   print_every,
                   agnc):
    def print_update(q):
        print("Function value:", q)

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    inlr, max_iter = epsilon, max_iter
    beta_1, beta_2 = 0.9, 0.999
    epsilon = 1e-08
    for t in range(max_iter):
        q = fun(x0)
        lr = decay_lr(inlr, 0.0001, t)
        # gradient norm clip will be implemented here.
        dq = grad(fun)(x0)
        norm = np.linalg.norm(dq)
        if norm > agnc:
            dq = dq / norm
        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t + 1))
        v_hat = v / (1 - np.power(beta_2, t + 1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        if t % print_every == 0:
            print_update(q)

    return x0




