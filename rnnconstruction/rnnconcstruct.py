
from fixedpointfinder.build_utils import build_rnn_ds, build_gru_ds
import autograd.numpy as np
from fixedpointfinder.minimization import adam_weights_optimizer
import tensorflow as tf
from rnnconstruction.build_utils import build_rnn_inducer, build_gru_inducer



class Rnnconstructor:

    adam_default_hps = {'alr_hps': {'decay_rate': 0.0001},
                        'agnc_hps': {'norm_clip': 1.0,
                                     'decay_rate': 1e-03},
                        'adam_hps': {'epsilon': 1e-02,
                                     'max_iters': 2000,
                                     'method': 'joint',
                                     'print_every': 200}}

    def __init__(self, fps, n_hidden, rnn_type,
                 alr_decayr=adam_default_hps['alr_hps']['decay_rate'],
                 agnc_normclip=adam_default_hps['agnc_hps']['norm_clip'],
                 agnc_decayr=adam_default_hps['agnc_hps']['decay_rate'],
                 epsilon=adam_default_hps['adam_hps']['epsilon'],
                 max_iters=adam_default_hps['adam_hps']['max_iters'],
                 method=adam_default_hps['adam_hps']['method'],
                 print_every=adam_default_hps['adam_hps']['print_every']):
        """This class has functionality to construct an RNN and optimize the
        recurrent kernel such that it can represent a given set of fixedpoints.
        The recurrent kernel may afterwars be reimplemented into a neural network
        with a specific recurrent layer. """
        self.n_hidden = n_hidden
        self.rnn_type = rnn_type
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
            target = np.empty((len(fps), self.n_hidden))
            for i in range(len(fps)):
                target[i, :] = fps[i]['x']
            return target

        target = create_target(self.fps)
        if self.rnn_type == 'vanilla':
            fun = build_rnn_inducer(target)
        elif self.rnn_type == 'gru':
            fun = build_gru_inducer(target, self.n_hidden)

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

    def compute_jacobians(self, fps, weights, inputs):

        fph = fps
        for i in range(len(fps)):
            if self.rnn_type == 'vanilla':
                fun, jac_fun = build_rnn_ds(weights, self.n_hidden, inputs[i, :], 'sequential')
            elif self.rnn_type == 'gru':
                fun, jac_fun = build_gru_ds(weights, self.n_hidden, inputs[i, :], 'sequential')
            fph[i]['jac'] = jac_fun(fps[i]['x'])
            fph[i]['fun'] = fun(fps[i]['x'])

        return fph


