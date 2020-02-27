

import autograd.numpy as np
from fixedpointfinder.minimization import adam_weights_optimizer
import tensorflow as tf

class Rnnconstructor():
    adam_default_hps = {'alr_hps': {'decay_rate': 0.0001},
                        'agnc_hps': {'norm_clip': 1.0,
                                     'decay_rate': 1e-03},
                        'adam_hps': {'epsilon': 1e-03,
                                     'max_iters': 2,
                                     'method': 'joint',
                                     'print_every': 200}}

    def __init__(self, fps,
                 alr_decayr=adam_default_hps['alr_hps']['decay_rate'],
                 agnc_normclip=adam_default_hps['agnc_hps']['norm_clip'],
                 agnc_decayr=adam_default_hps['agnc_hps']['decay_rate'],
                 epsilon=adam_default_hps['adam_hps']['epsilon'],
                 max_iters=adam_default_hps['adam_hps']['max_iters'],
                 method=adam_default_hps['adam_hps']['method'],
                 print_every=adam_default_hps['adam_hps']['print_every']):

        self.alr_decayr = alr_decayr
        self.agnc_normclip = agnc_normclip
        self.agnc_decayr = agnc_decayr
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.method = method
        self.print_every = print_every

        self.verbose = True
        self.fps = fps
        self.fun = self.build_model()

    def build_model(self):

        def create_target(fps):
            target = np.empty((len(fps), 24))
            for i in range(len(fps)):
                target[i, :] = fps[i]['x']
            return target

        target = create_target(self.fps)
        def fun(x):
            return np.max(0.5 * np.sum(((- target + np.matmul(np.tanh(target), x)) ** 2), axis=1))

        return fun

    def train_recurrentweights(self, weights):

        weights = adam_weights_optimizer(self.fun, weights, 0,
                                         epsilon=self.epsilon,
                                         alr_decayr=self.alr_decayr,
                                         max_iter=self.max_iters,
                                         print_every=self.print_every,
                                         init_agnc=self.agnc_normclip,
                                         agnc_decayr=self.agnc_decayr,
                                         verbose=self.verbose)

        return weights

