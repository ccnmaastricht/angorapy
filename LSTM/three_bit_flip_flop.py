import tensorflow as tf
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from utilities.model_management import build_sub_model_to
from mpl_toolkits.mplot3d import Axes3D
from LSTM.fixedpointfinder import FixedPointFinder, FPF_adam
from tensorflow.keras.models import load_model
import os

class Flipflopper:
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
                    'model_name': 'flipflopmodel'}
        self.data_hps = {'n_batch': 128,
                         'n_time': 256,
                         'n_bits': 3,
                         'p_flip': 0.2}
        # data_hps may be changed but are recommended to remain at their default values

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
            x = tf.keras.layers.SimpleRNN(n_hidden, name=self.hps['rnn_type'], return_sequences=True, stateful=True)(inputs)
        elif self.hps['rnn_type'] == 'gru':
            x = tf.keras.layers.GRU(n_hidden, name=self.hps['rnn_type'], return_sequences=True, stateful=True)(inputs)
        elif self.hps['rnn_type'] == 'lstm':
            x = tf.keras.layers.LSTM(n_hidden, name=self.hps['rnn_type'], return_sequences=True)(inputs)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        x = tf.keras.layers.Dense(3)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        self.model.summary()


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


        self.rng = npr.RandomState(125)
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

    def train(self, stim):
        '''Function to train an RNN model and visualize training history.

        Args:
            stim: dict containing 'inputs' and 'output' as input and target data for training the model.

                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.
                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the FlipFlop memory device.

        Returns:
            None.'''
        self._build_model()
        self.model.compile(optimizer="adam", loss="mse",
                  metrics=['accuracy'])
        self.history = self.model.fit(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32),
                            tf.convert_to_tensor(stim['output'], dtype=tf.float32), epochs=4000)

        self._save_model()

        plt.figure()
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.title('Training history')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

    def visualize_flipflop(self, stim):
        prediction = self.model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
        fig.suptitle('3-Bit Flip-Flop using '+ self.hps['rnn_type'])
        ax1.plot(prediction[0, :, 0], c='r')
        ax1.plot(stim['inputs'][0, :, 0], c='k')
        ax2.plot(stim['inputs'][0, :, 1], c='k')
        ax2.plot(prediction[0, :, 1], c='g')
        ax3.plot(stim['inputs'][0, :, 2], c='k')
        ax3.plot(prediction[0, :, 2], c='b')
        plt.yticks([-1, +1])
        plt.xlabel('Time')
        ax1.xaxis.set_visible(False)
        ax2.xaxis.set_visible(False)

        plt.show()
# TODO: both visualization of training history and visualization of flipflop need improvement.

    def find_fixed_points(self, stim):
        '''Intialize class FixedPointFinder'''

        self._load_model()
        self._get_activations(stim)
        self.hps = {'rnn_type': self.hps['rnn_type'],
                    'n_hidden': self.hps['n_hidden'],
                    'unique_tol': 1e-03,
                    'threshold': 1e-10,
                    'n_ic': 20,
                    'algorithm': "adam",
                    'scipy_hps': {'method': "Newton-CG",
                                  'display': True},
                    'adam_hps': {'max_iter': 5000,
                                 'lr': 0.001,
                                 'n_init': 4,
                                 'gradientnormclip': 1.0,
                                 'print_every': 200}}
        weights = self.model.get_layer(self.hps['rnn_type']).get_weights()
        return weights
        self.finder = FixedPointFinder(weights, self.hps['rnn_type'])
        self.ffinder = FPF_adam()

    def _save_model(self):
        '''Save trained model to JSON file.'''
        self.model.save(os.getcwd()+"/saved/"+self.hps['rnn_type']+"model.h5")
        print("Saved "+self.hps['rnn_type']+" model.")

    def _load_model(self):
        """Load saved model from JSON file.
        The function will overwrite the current model, if it exists."""
        self.model = load_model(os.getcwd()+"/saved/"+self.hps['rnn_type']+"model.h5")
        print("Loaded "+self.hps['rnn_type']+" model.")

    def _get_activations(self, stim):
        sub_model = build_sub_model_to(self.model, [self.hps['rnn_type']])
        self.activation = sub_model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))






if __name__ == "__main__":
    rnn_type = 'gru'
    n_hidden = 24

    flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
    stim = flopper.generate_flipflop_trials()

    # flopper.train(stim)

    # flopper.visualize_flipflop(stim)

    weights = flopper.find_fixed_points(stim)


