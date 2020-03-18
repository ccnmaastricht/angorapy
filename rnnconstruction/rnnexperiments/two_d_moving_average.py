import numpy as np
import tensorflow as tf
from utilities.model_management import build_sub_model_to
import matplotlib.pyplot as plt

class Avemover:

    def __init__(self, N, seconds):
        self.N = N
        self.seconds = seconds


    def gen_data(self, nPulses: int = 100):
        delta_t = 0.1
        seconds = self.seconds
        nPulses = nPulses
        targetpulse = int(1/delta_t)
        lenP = int(1/delta_t) # one second
        x = np.random.uniform(-1, 1, nPulses)
        pulses = np.reshape(np.repeat(x, lenP), (nPulses, lenP)).transpose()

        pulse_beginnings = np.random.randint(0, seconds, nPulses)
        pulse_beginnings.sort()
        # pulse_beginnings = np.append(pulse_beginnings, seconds)

        input = np.zeros((int(seconds/delta_t), 1, 1))
        target = np.zeros(int(seconds/delta_t))
        # target[0:targetpulse] = np.zeros(targetpulse)
        for i in range(pulse_beginnings.shape[0]):
            input[int(pulse_beginnings[i]/delta_t):(int(pulse_beginnings[i]/delta_t)+lenP), 0, 0] = pulses[:, i]
            # time_difference = pulse_beginnings[i] - pulse_beginnings[i - 1]
            target[int(pulse_beginnings[i] / delta_t - pulse_beginnings[i-1] / delta_t):int(pulse_beginnings[i] / delta_t)] \
                = np.mean((pulses[0, i-1], pulses[0, (i - 2)]))# , int(pulse_beginnings[i+1] / delta_t - pulse_beginnings[i] / delta_t))
            # if time_difference <= 30 & time_difference >= 5:
            #input[int(pulse_beginnings[i+1]/delta_t):(int(pulse_beginnings[i+1]/delta_t)+lenP), 0, 1] = np.ones(lenP)


        return tf.convert_to_tensor(input, dtype=tf.float32), tf.convert_to_tensor(target, dtype=tf.float32)

    def generate_data(self, delta_t, npulses: int = 100):
        seconds = self.seconds
        pulse_length = int(1/delta_t)
        pulses = np.random.uniform(-1, 1, npulses)

        pulse_beginnings = np.random.randint(0, seconds, npulses)
        pulse_beginnings.sort()
        input = np.empty((int(seconds/delta_t), 1, 1))
        target = np.empty(int(seconds/delta_t))

        for i in range(npulses-1):
            input[int(pulse_beginnings[i] / delta_t):int(pulse_beginnings[i] / delta_t + pulse_length)] = pulses[i]
            if i == 0:
                target[int(pulse_beginnings[i] / delta_t):int(pulse_beginnings[i] / delta_t+ pulse_length)] = pulses[i]
            else:
                target[int(pulse_beginnings[i] / delta_t):int(pulse_beginnings[i] / delta_t+ pulse_length)] \
                    = np.mean((pulses[i], pulses[i - 1]))

        return tf.convert_to_tensor(input, dtype=tf.float32), tf.convert_to_tensor(target, dtype=tf.float32)


    def build_model(self, inputshape):
        name = "Averagemover"

        inputs = tf.keras.Input(shape=inputshape, name="input")

        x = tf.keras.layers.SimpleRNN(self.N, name='rnn')(inputs)
        # x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Dense(1, activation="linear", name="dense")(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        model.summary()

        return model


if __name__ == "__main__":

    mover = Avemover(N=100, seconds=100)
    input, target = mover.generate_data(npulses=20, delta_t=0.01)
    model = mover.build_model((1, 1))

    model.compile(optimizer="adam", loss="mse",
                  metrics=['accuracy'])

    history = model.fit(input, target, epochs=5)

    prediction = model.predict(input)

    weights = layer = model.get_layer('rnn').get_weights()
    sub_model = build_sub_model_to(model, ['input', 'rnn'])
    sub_model.summary()

    activations = sub_model.predict(input)
    print(activations[1])
    plt.plot(activations[1])
    plt.title('activations')
    plt.show()

    plt.plot(target.numpy())
    plt.title('target')
    plt.show()
    input = input.numpy()
    plt.plot(input[:, :, 0])
    plt.title('input')
    plt.show()

    # plt.plot(input[:, 0, :])
    # plt.show()




