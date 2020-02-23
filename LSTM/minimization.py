import autograd.numpy as np
from autograd import grad

def adam_optimizer(fun, x0,
                   epsilon, alr_decayr,
                   max_iter, print_every,
                   init_agnc, agnc_decayr,
                   verbose):
    """Function to implement the adam optimization algorithm. Also included in this function are
    functionality for adaptive learning rate as well as adaptive gradient norm clipping."""
    def print_update(q, lr):
        print("Function value:", q, "; lr:", np.round(lr, 4))

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    def decay_agnc(initial_clip, decay, iteration):
        return initial_clip * (1.0 / (1.0 + decay * iteration))

    beta_1, beta_2 = 0.9, 0.999
    eps = 1e-08
    m, v = np.zeros(x0.shape), np.zeros(x0.shape)
    for t in range(max_iter):
        q = fun(x0)
        lr = decay_lr(epsilon, alr_decayr, t)
        agnc = decay_agnc(init_agnc, agnc_decayr, t)

        dq = grad(fun)(x0)
        norm = np.linalg.norm(dq)
        if norm > agnc:
            dq = dq / norm
        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t + 1))
        v_hat = v / (1 - np.power(beta_2, t + 1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + eps)

        if t % print_every == 0 and verbose:
            print_update(q, lr)

    return x0


def adam_weights_optimizer(fun, x0, mean_vel,
                           epsilon, alr_decayr,
                           max_iter, print_every,
                           init_agnc, agnc_decayr,
                           verbose):
    """Function to implement the adam optimization algorithm. Also included in this function are
    functionality for adaptive learning rate as well as adaptive gradient norm clipping."""
    def print_update(q, lr):
        print("Function value:", q, "; lr:", np.round(lr, 4))

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    def decay_agnc(initial_clip, decay, iteration):
        return initial_clip * (1.0 / (1.0 + decay * iteration))

    beta_1, beta_2 = 0.9, 0.999
    eps = 1e-08
    m, v = np.zeros(x0.shape), np.zeros(x0.shape)
    for t in range(max_iter):
        q = fun(x0)
        lr = decay_lr(epsilon, alr_decayr, t)
        agnc = decay_agnc(init_agnc, agnc_decayr, t)

        dq = grad(fun)(x0)
        dq = (q - mean_vel) * dq
        norm = np.linalg.norm(dq)
        if norm > agnc:
            dq = dq / norm
        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t + 1))
        v_hat = v / (1 - np.power(beta_2, t + 1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + eps)

        if t % print_every == 0 and verbose:
            print_update((q - mean_vel), lr)

    return x0

def adam_lstm(h_fun, c_fun, x0,
                   epsilon, alr_decayr,
                   max_iter, print_every,
                   init_agnc, agnc_decayr,
                   verbose):
    """Function to implement the adam optimization algorithm. Also included in this function are
    functionality for adaptive learning rate as well as adaptive gradient norm clipping."""
    def print_update(q, qc, lr):
        print("Function value:", q, "; c_fun:",qc, "; lr", np.round(lr, 4))

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    def decay_agnc(initial_clip, decay, iteration):
        return initial_clip * (1.0 / (1.0 + decay * iteration))

    n_hidden = int(len(x0)/2)
    c, h = x0[n_hidden:], x0[:n_hidden]

    beta_1, beta_2 = 0.9, 0.999
    eps = 1e-08
    m, v = np.zeros(x0.shape), np.zeros(x0.shape)
    for t in range(max_iter):
        q = h_fun(x0)
        qc = c_fun(x0)
        lr = decay_lr(epsilon, alr_decayr, t)
        agnc = decay_agnc(init_agnc, agnc_decayr, t)

        dh = grad(h_fun)(x0)
        dc = grad(c_fun)(x0)
        norm_h = np.linalg.norm(dh)
        norm_c = np.linalg.norm(dc)
        if norm_h > agnc:
            dh = dh / norm_h
        if norm_c > agnc:
            dc = dc / norm_c

        dq = dh
        # dq = np.vstack((dh, dc))
        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t + 1))
        v_hat = v / (1 - np.power(beta_2, t + 1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + eps)

        if t % print_every == 0 and verbose:
            print_update(q, qc, lr)

    return x0


