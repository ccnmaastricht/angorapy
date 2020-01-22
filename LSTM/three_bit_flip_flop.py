import tensorflow as tf
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

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

    Hyperparameters:
    '''

    def __init__(self):
        self.hps = {'rnn_type': 'lstm',
                    'n_hidden': 24,
                    'model_name': 'flipflopmodel'}
        self.data_hps = {'n_batch': 128,
                         'n_time': 256,
                         'n_bits': 3,
                         'p_flip': 0.2}


    def build_model(self):
        n_hidden = self.hps['n_hidden']
        name = self.hps['model_name']
        # inputshape=(self.data_hps['n_time'], self.data_hps['n_bits'])
        n_time, n_batch, n_bits = self.data_hps['n_time'], self.data_hps['n_batch'], self.data_hps['n_bits']

        inputs = tf.keras.Input(shape=(n_time, n_bits), batch_size=n_batch)

        if self.hps['rnn_type'] == 'vanilla':
            x = tf.keras.layers.SimpleRNN(n_hidden, name='vanilla', return_sequences=True)(inputs)
            x = tf.keras.layers.Dense(3)(x)
        elif self.hps['rnn_type'] == 'gru':
            x = tf.keras.layers.GRU(n_hidden, name='gru', return_sequences=True)(inputs)
            x = tf.keras.layers.Dense(3)(x)
        elif self.hps['rnn_type'] == 'lstm':
            x = tf.keras.layers.LSTM(n_hidden, name='lstm', return_sequences=True)(inputs)
            x = tf.keras.layers.Dense(3)(x)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        self.model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        self.model.summary()



    def generate_stim(self):
        batch_size = self.data_hps['n_batch']
        pulse_duration = 10
        bits = self.data_hps['n_bits']
        nPulses = np.floor(self.data_hps['n_time']*1)
        input = np.zeros((batch_size, self.data_hps['n_time'], bits))
        T = np.arange(0, size, int(size/nPulses))

        tstart = np.random.permutation(T)
        for i in range(int(nPulses)):
            if tstart[i] <= (size-pulse_duration):
                input[tstart[i]:tstart[i]+pulse_duration, 0, np.random.randint(0, 3)] = 2*(np.random.rand() < .5)-1

        # generate output
        target = np.zeros((size, bits))

        prev = np.zeros((batch_size, bits))
        for t in range(size-batch_size):
            changed = np.argwhere(input[t, :, :])
            prev[0, changed[0, 1]] = input[t, 0, changed[0, 1]]
            target[t, :] = prev


        return input, target

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
                the correct behavior of the FlipFlop memory device.
        '''
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
        self.model.compile(optimizer="adam", loss="mse",
                  metrics=['accuracy'])
        self.history = self.model.fit(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32),
                            tf.convert_to_tensor(stim['output'], dtype=tf.float32), epochs=2000)

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




if __name__ == "__main__":
    flopper = Flipflopper()
    stim = flopper.generate_flipflop_trials()

    flopper.build_model()
    flopper.train(stim)

    flopper.visualize_flipflop(stim)


