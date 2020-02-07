import autograd.numpy as np
import numdifftools as nd
from autograd import grad



def backproprnn(combined):

    def print_update(q):
        print("Function value:", q)

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    x0, input, weights, n_hidden = combined[0], combined[1], combined[2], combined[3]

    weights, inputweights, b = weights[1], weights[0], weights[2]
    projection_b = np.matmul(input, inputweights) + b
    fun = lambda x: np.mean(0.5 * np.sum(np.square(- x + np.matmul(np.tanh(x), weights) + b), axis=1))
    # grad_fun = nd.Gradient(fun)
    inlr, max_iter = 0.01, 7000
    beta_1, beta_2 = 0.9, 0.999
    m, v = np.zeros(n_hidden), np.zeros(n_hidden)
    epsilon = 1e-08
    for t in range(max_iter):
        q = fun(x0)
        lr = decay_lr(inlr, 0.0001, t)
        # gradient norm clip will be implemented here.
        dq = grad(fun)(x0)
        norm = np.linalg.norm(dq)
        if norm > 1.0:
            dq = dq/norm
        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t+1))
        v_hat = v / (1 - np.power(beta_2, t+1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        if t % 200 == 0:
            print_update(q)

    print('new IC')
    fixedpoints = []
    jac_fun = lambda x: - np.eye(n_hidden, n_hidden) + weights * (1 - np.tanh(x) ** 2)
    for i in range(5):
        jacobian = jac_fun(x0[i, :])
        fixedpoint = {'fun': q,
                      'x': x0[i, :],
                      'jac': jacobian}
        fixedpoints.append(fixedpoint)

    return fixedpoints

def backpropgru(combined):


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def print_update(q):
        print("Function value:", q, ";")

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))


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

    z_fun = lambda x: sigmoid(np.matmul(x, U_z) + b_z)
    r_fun = lambda x: sigmoid(np.matmul(x, U_r) + b_r)
    g_fun = lambda x: np.tanh((r_fun(x) * np.matmul(x, U_h) + b_h))

    fun = lambda x: np.mean(0.5 * sum(((1 - z_fun(x)) * (g_fun(x) - x)) ** 2))
    grad_fun = nd.Gradient(fun)

    inlr, max_iter = 0.001, 6000
    beta_1, beta_2 = 0.9, 0.999
    m, v = np.zeros(n_hidden), np.zeros(n_hidden)
    epsilon = 1e-08
    for t in range(max_iter):
        q = fun(x0)
        # dq = grad_fun(x0)
        lr = decay_lr(inlr, 0.0001, t)
        dq = grad(fun)(x0)
        norm = np.linalg.norm(dq)
        if norm > 1.0:
            dq = dq/norm

        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t+1))
        v_hat = v / (1 - np.power(beta_2, t+1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        if t % 200 == 0:
            print_update(q)

    dynamical_system = lambda x: (1 - z_fun(x[0:n_hidden])) * (g_fun(x[0:n_hidden]) - x[0:n_hidden])
    jac_fun = nd.Jacobian(dynamical_system)
    fixedpoints = []
    for i in range(5):
        jacobian = jac_fun(x0[i, :])
        fixedpoint = {'fun': q,
                      'x': x0[i, :],
                      'jac': jacobian}
        fixedpoints.append(fixedpoint)

    return fixedpoints




