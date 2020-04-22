import numpy as np
from copy import copy
from collections import OrderedDict
import os
import gym
import matplotlib.pyplot as plt
from utilities.util import parse_state

from analysis.chiefinvestigation import Chiefinvestigator
import sklearn.decomposition as skld

EPSILON = 1e-6


class reach_agent:

    def __init__(self, weights_biases, env, h0, means, variances, n_hidden = 32):
        self.softplus = lambda x: np.log(np.exp(x) + 1)
        self.sigma_g = lambda x: 1 / (1 + np.exp(-x))
        self.sigma_h = lambda x: np.tanh(x)

        if not env == gym.Env:
            self.env = gym.make(env)
        else:
            self.env = env
        self.state = self.env.reset()
        self.n_hidden = n_hidden
        self.h0 = h0
        self.h = copy(self.h0)
        self.input_proj = np.zeros(n_hidden)

        self.means = means
        self.variances = variances

        self.__upstream__(weights_biases)
        self.__downstream__(weights_biases)
        self.__recurrent__(weights_biases)

    def __upstream__(self, weights_biases):
        self.ff_layer = OrderedDict()
        self.ff_layer['one'] = lambda x: np.tanh(x @ weights_biases[0] + weights_biases[1])
        self.ff_layer['two'] = lambda x: np.tanh(x @ weights_biases[2] + weights_biases[3])
        self.ff_layer['three'] = lambda x: np.tanh(x @ weights_biases[4] + weights_biases[5])
        self.ff_layer['four'] = lambda x: np.tanh(x @ weights_biases[6] + weights_biases[7])

    def __downstream__(self, weights_biases):
        self.alpha_fun = lambda x: self.softplus(x @ weights_biases[11] + weights_biases[12])
        self.beta_fun = lambda x: self.softplus(x @ weights_biases[13] + weights_biases[14])

    def __recurrent__(self, weights_biases):
        z, r, h = np.arange(0, self.n_hidden), np.arange(self.n_hidden, 2 * self.n_hidden), np.arange(2 * self.n_hidden, 3 * self.n_hidden)
        W_z, W_r, W_h = weights_biases[8][:, z], weights_biases[8][:, r], weights_biases[8][:, h]
        U_z, U_r, U_h = weights_biases[9][:, z], weights_biases[9][:, r], weights_biases[9][:, h]
        b_z, b_r, b_h = weights_biases[10][0, z], weights_biases[10][0, r], weights_biases[10][0, h]
        self.gru_layer = OrderedDict()
        self.gru_layer['z'] = lambda x, h: self.sigma_g(x @ W_z + h @ U_z + b_z)
        self.gru_layer['r'] = lambda x, h: self.sigma_g( x @ W_r + h @ U_r + b_r)
        self.gru_layer['g'] = lambda x, r, h: self.sigma_g( x @ W_h + (r * h) @ U_h + b_h)

    def __dh__(self):
        alpha = self.alpha_fun(self.h)
        beta = self.beta_fun(self.h)

        action = (alpha - 1) / (alpha + beta - 2)

        observation, reward, done, info = self.env.step(action)

        self.state = (parse_state(observation) - self.means) / (np.sqrt(self.variances) + EPSILON)
        x = self.state
        for key in self.ff_layer:
            x = self.ff_layer[key](x)

        self.input_proj = copy(x)
        z = self.gru_layer['z'](x, self.h)
        r = self.gru_layer['r'](x, self.h)
        g = self.gru_layer['g'](x, r, self.h)
        self.dh = (1 - z) * (g - self.h)

    def update(self, dt=1e-3):
        self.__dh__()
        self.h += dt * self.dh

    def compute_q(self):
        self.__dh__()
        return self.dh @ self.dh.transpose()

    def reset(self):
        observation = self.env.reset()

        self.state = (parse_state(observation) - self.means) / (np.sqrt(self.variances) + EPSILON)
        self.h = copy(self.h0)

    def set_h0(self, h0):
        self.h0 = h0
        self.h = copy(h0)


if __name__ == "__main__":
    os.chdir("../")  # remove if you want to search for ids in the analysis directory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    agent_id, env = 1585777856, "HandFreeReachLFAbsolute-v0" # free reach
    # agent_id = 1587117437 # free reach relative control
    chiefinvesti = Chiefinvestigator(agent_id)
    means = chiefinvesti.preprocessor.wrappers[0].mean[0]
    variances = chiefinvesti.preprocessor.wrappers[0].variance[0]

    layer_names = chiefinvesti.get_layer_names()
    activations_single_run, inputs_single_run, actions_single_run = chiefinvesti.get_data_over_single_run('policy_recurrent_layer',
                                                                                                          layer_names[1])
    np.random.seed(420)
    h0 = np.random.random(32) * 2 - 1
    reacher = reach_agent(chiefinvesti.network.get_weights(), env, h0, means, variances,
                          n_hidden=chiefinvesti.n_hidden)

    reacher.env.sim.nsubsteps = 5
    dt = 1e-2
    t_sim = 150
    t_steps = int(t_sim / dt) + 1

    h_vector = np.zeros((t_steps, 32))
    dh_vector = np.zeros_like(h_vector)
    input_projections = np.zeros_like(h_vector)
    q_vector = np.zeros(t_steps)
    dV = np.zeros(t_steps)
    reacher.reset()
    for t in range(t_steps):
        h_vector[t, :] = reacher.h
        q_vector[t] = reacher.compute_q()
        input_projections[t, :] = reacher.input_proj
        dh_vector[t, :] = reacher.dh
        dV[t] = np.sum(2 * reacher.h * reacher.dh)
        reacher.update(dt=dt)

    fig= plt.figure(1)
    plt.subplot(121)
    plt.plot(h_vector)
    plt.subplot(122)
    plt.plot(q_vector)
    plt.show()

    q_threshold = 1e-2
    q_smaller_threshold = q_vector[q_vector < q_threshold]
    random_index = np.random.randint(q_smaller_threshold.shape[0], size=1)
    random_index = np.argmin(q_vector)

    pca = skld.PCA(2)
    transformed_h_vector = pca.fit_transform(h_vector)

    center_point = transformed_h_vector[random_index, :]

    sidesteps, n_points = 3, 25
    left_points = center_point.T + center_point.T * sidesteps
    right_points = center_point.T - center_point.T * sidesteps

    x,y = np.meshgrid(np.linspace(right_points[0], left_points[0], n_points),
                        np.linspace(right_points[1], left_points[1], n_points),
                        indexing='xy')
    x,y = x.ravel(), y.ravel()

    stacked_points = np.vstack((x, y)).T
    inverse_stacked_points = pca.inverse_transform(stacked_points)


    h_vector_mesh = np.zeros_like(inverse_stacked_points)
    dVn = np.zeros(len(inverse_stacked_points))
    reacher.reset()
    for i in range(len(inverse_stacked_points)):
        reacher.reset()
        reacher.h = inverse_stacked_points[i, :]
        reacher.update(dt=dt)
        h_vector_mesh[i, :] = reacher.h
        dVn[i] = np.sum(2 * (reacher.h - h_vector[random_index, :]) * reacher.dh)

    transformed_h_mesh = pca.transform(h_vector_mesh)
    dVn_r = dVn.reshape((n_points,n_points)) < 0
    #plt.imshow(dVn_r, extent=[min(x), max(x), min(y), max(y)])

    plt.scatter(center_point[0], center_point[1], c='r')
    plt.plot(transformed_h_vector[:, 0], transformed_h_vector[:, 1], c='g')
   # plt.quiver(stacked_points[:, 0], stacked_points[:, 1],
     #         transformed_h_mesh[:, 0], transformed_h_mesh[:, 1])



    plt.imshow(dVn.reshape(n_points, n_points)<0, extent=[min(stacked_points[:, 0]),
                                                          max(stacked_points[:, 0]),
                                                          min(stacked_points[:, 1]),
                                                          max(stacked_points[:, 1])])
    plt.show()




