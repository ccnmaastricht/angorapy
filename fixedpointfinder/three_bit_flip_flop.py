import tensorflow as tf
import numpy as np
import numpy.random as npr
from utilities.model_utils import build_sub_model_to
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.layers import SimpleRNNCell, SimpleRNN
from tensorflow.python.keras.utils import tf_utils

class Flipflopper(object):
    ''' Class for training an RNN to implement a 3-Bit Flip-Flop task as
    described in Sussillo, D., & Barak, O. (2012). Opening the Black Box:
    Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks.
    Neural Computation, 25(3), 626–649. https://doi.org/10.1162/NECO_a_00409

    Task:
        A set of three inputs submit transient pulses of either -1 or +1. The
        task of the model is to return said inputs until one of them flips.
        If the input pulse has the same value as the previous input in a given
        channel, the output must not change. Thus, the RNN has to memorize
        previous input pulses. The number of input channels is not limited
        in theory.

    Usage:
        The class Flipflopper can be used to build and train a model of type
        RNN on the 3-Bit Flip-Flop task. Furthermore, the class can make use
        of the class FixedPointFinder to analyze the trained model.

    Hyperparameters:
        rnn_type: Specifies architecture of type RNN. Must be one of
        ['vanilla','gru', 'lstm']. Will raise ValueError if
        specified otherwise. Default is 'vanilla'.

        n_hidden: Specifies the number of hidden units in RNN. Default
        is: 24.

    '''

    def __init__(self, rnn_type: str = 'vanilla', n_hidden: int = 24):

        self.hps = {'rnn_type': rnn_type,
                    'n_hidden': n_hidden,
                    'model_name': 'flipflopmodel',
                    'verbose': False}
        self.data_hps = {'n_batch': 128,
                         'n_time': 256,
                         'n_bits': 3,
                         'p_flip': 0.3}
        self.verbose = self.hps['verbose']
        # data_hps may be changed but are recommended to remain at their default values
        self.model, self.weights = self._build_model()
        self.rng = npr.RandomState(125)

    def _build_model(self):
        '''Builds model that can be used to train the 3-Bit Flip-Flop task.

        Args:
            None.

        Returns:
            None.'''
        n_hidden = self.hps['n_hidden']
        name = self.hps['model_name']

        n_time, n_batch, n_bits = self.data_hps['n_time'], self.data_hps['n_batch'], self.data_hps['n_bits']

        inputs = tf.keras.Input(shape=(n_time, n_bits), batch_size=n_batch, name='input')

        if self.hps['rnn_type'] == 'vanilla':
            x = tf.keras.layers.SimpleRNN(n_hidden, name=self.hps['rnn_type'], return_sequences=True)(inputs)
        elif self.hps['rnn_type'] == 'gru':
            x = tf.keras.layers.GRU(n_hidden, name=self.hps['rnn_type'], return_sequences=True)(inputs)
        elif self.hps['rnn_type'] == 'lstm':
            x, state_h, state_c = tf.keras.layers.LSTM(n_hidden, name=self.hps['rnn_type'], return_sequences=True,
                                                       stateful=True, return_state=True,
                                                       implementation=1)(inputs)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        x = tf.keras.layers.Dense(3)(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        weights = model.get_layer(self.hps['rnn_type']).get_weights()

        if self.verbose:
            model.summary()

        return model, weights


    def generate_flipflop_trials(self):
        '''Generates synthetic data (i.e., ground truth trials) for the
        FlipFlop task. See comments following FlipFlop class definition for a
        description of the input-output relationship in the task.

        Args:
            None.
        Returns:
            dict containing 'inputs' and 'outputs'.
                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.
                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the FlipFlop memory device.'''
        data_hps = self.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        n_bits = data_hps['n_bits']
        p_flip = data_hps['p_flip']

        # Randomly generate unsigned input pulses
        unsigned_inputs = self.rng.binomial(
            1, p_flip, [n_batch, n_time, n_bits])

        # Ensure every trial is initialized with a pulse at time 0
        unsigned_inputs[:, 0, :] = 1

        # Generate random signs {-1, +1}
        random_signs = 2 * self.rng.binomial(
            1, 0.5, [n_batch, n_time, n_bits]) - 1

        # Apply random signs to input pulses
        inputs = np.multiply(unsigned_inputs, random_signs)

        # Allocate output
        output = np.zeros([n_batch, n_time, n_bits])

        # Update inputs (zero-out random start holds) & compute output
        for trial_idx in range(n_batch):
            for bit_idx in range(n_bits):
                input_ = np.squeeze(inputs[trial_idx, :, bit_idx])
                t_flip = np.where(input_ != 0)
                for flip_idx in range(np.size(t_flip)):
                    # Get the time of the next flip
                    t_flip_i = t_flip[0][flip_idx]

                    '''Set the output to the sign of the flip for the
                    remainder of the trial. Future flips will overwrite future
                    output'''
                    output[trial_idx, t_flip_i:, bit_idx] = \
                        inputs[trial_idx, t_flip_i, bit_idx]

        return {'inputs': inputs, 'output': output}

    def train(self, stim, epochs, save_model: bool = True):
        '''Function to train an RNN model This function will save the trained model afterwards.

        Args:
            stim: dict containing 'inputs' and 'output' as input and target data for training the model.

                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.
                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the FlipFlop memory device.

        Returns:
            None.'''

        self.model.compile(optimizer="adam", loss="mse",
                  metrics=['accuracy'])
        history = self.model.fit(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32),
                            tf.convert_to_tensor(stim['output'], dtype=tf.float32), epochs=epochs)
        if save_model:
            self._save_model()
        return history

    def _save_model(self):
        '''Save trained model to JSON file.'''
        self.model.save(os.getcwd()+"/saved/"+self.hps['rnn_type']+"model.h5")
        print("Saved "+self.hps['rnn_type']+" model.")

    def load_model(self):
        """Load saved model from JSON file.
        The function will overwrite the current model, if it exists."""
        self.model = load_model(os.getcwd()+"/saved/"+self.hps['rnn_type']+"model.h5")
        print("Loaded "+self.hps['rnn_type']+" model.")

    def get_activations(self, stim):
        sub_model = build_sub_model_to(self.model, [self.hps['rnn_type']])
        activation = sub_model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))

        return activation


class PretrainableFlipflopper(Flipflopper):

    def __init__(self, rnn_type: str = 'vanilla', n_hidden: int = 24):

        super(Flipflopper, self).__init__(rnn_type, n_hidden)

    def pretrained_model(self, weights, recurrentweights, recurrent_trainable=False):

        '''Builds model that can be used to train the 3-Bit Flip-Flop task. A pretrained
        model assumes that the recurrent kernel has been optimized in some way. Thus, it
        offers functionality to not train the recurrent kernel furthermore.

        Args:
            None.

        Returns:
            None.'''

        n_hidden = self.hps['n_hidden']
        name = self.hps['model_name']
        n_time, n_batch, n_bits = self.data_hps['n_time'], self.data_hps['n_batch'], self.data_hps['n_bits']

        inputs = tf.keras.Input(shape=(n_time, n_bits), batch_size=n_batch, name='input')

        if self.hps['rnn_type'] == 'vanilla':
            x = SimplerRNN(n_hidden, name=self.hps['rnn_type'], return_sequences=True,
                           recurrent_trainable=recurrent_trainable)(inputs)
        elif self.hps['rnn_type'] == 'gru':
            x = tf.keras.layers.GRU(n_hidden, name=self.hps['rnn_type'], return_sequences=True)(inputs)
        elif self.hps['rnn_type'] == 'lstm':
            x, state_h, state_c = tf.keras.layers.LSTM(n_hidden, name=self.hps['rnn_type'], return_sequences=True, stateful=True,
                                     return_state=True)(inputs)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        x = tf.keras.layers.Dense(3)(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)

        # weights = model.get_layer(self.hps['rnn_type']).get_weights()
        weights = self.build_pretrained_model(weights, recurrentweights, recurrent_trainable)
        model.get_layer(self.hps['rnn_type']).set_weights(weights)

        if self.verbose:
            model.summary()

        return model

    def build_pretrained_model(self, weights, recurrentweights, recurrent_trainable=False):

        inputweights = weights[0]
        recurrentbias = weights[2]

        if not recurrent_trainable:
            return [inputweights, recurrentbias, recurrentweights]
        else:
            return [inputweights, recurrentweights, recurrentbias]

    def train_pretrained(self, stim, epochs, weights, recurrentweights, recurrent_trainable):
        '''Function to train a pretrained RNN model.

        Args:
            stim: dict containing 'inputs' and 'output' as input and target data for training the model.

                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.
                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the FlipFlop memory device.

        Returns:
            None.'''

        model = self.pretrained_model(weights, recurrentweights, recurrent_trainable)
        model.compile(optimizer="adam", loss="mse",
                      metrics=['accuracy'])
        history = model.fit(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32),
                            tf.convert_to_tensor(stim['output'], dtype=tf.float32), epochs=epochs)

        return history

    @staticmethod
    def pretrained_predict(model , stim):

        model.compile(optimizer="adam", loss="mse",
                      metrics=['accuracy'])
        score = model.evaluate(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32),
                               tf.convert_to_tensor(stim['output'], dtype=tf.float32))

        return score


class SimplerRNN(SimpleRNN):

    def __init__(self, units,
                   activation='tanh',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   bias_initializer='zeros',
                   kernel_regularizer=None,
                   recurrent_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None,
                   kernel_constraint=None,
                   recurrent_constraint=None,
                   bias_constraint=None,
                   dropout=0.,
                   recurrent_dropout=0.,
                   return_sequences=False,
                   return_state=False,
                   go_backwards=False,
                   stateful=False,
                   unroll=False,
                recurrent_trainable=False,
                    **kwargs):
        cell = SimplerRNNCell(units,
                              activation=activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                recurrent_regularizer=recurrent_regularizer,
                                bias_regularizer=bias_regularizer,
                                kernel_constraint=kernel_constraint,
                                recurrent_constraint=recurrent_constraint,
                                bias_constraint=bias_constraint,
                                dropout=dropout,
                                recurrent_dropout=recurrent_dropout,
                              recurrent_trainable=recurrent_trainable,
                                dtype=kwargs.get('dtype'))
        super(SimpleRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        # self.activity_regularizer = regularizers.get(activity_regularizer)
        # self.input_spec = [InputSpec(ndim=3)]


class SimplerRNNCell(SimpleRNNCell):
    def __init__(self, output_dim, recurrent_trainable, **kwargs):
        self.output_dim = output_dim
        super(SimplerRNNCell, self).__init__(output_dim, **kwargs)
        self.recurrent_trainable = recurrent_trainable

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=self.recurrent_trainable)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True