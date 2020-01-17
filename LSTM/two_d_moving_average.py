import numpy as np
import tensorflow as tf

class Avemover:

    def __init__(self, N, seconds):
        self.N = N
        self.seconds = seconds


    def gen_data(self, nPulses: int = 100):
        delta_t = 0.01
        seconds = self.seconds
        nPulses = nPulses
        targetpulse = 25
        x = np.random.uniform(-1, 1, nPulses)
        pulses = np.reshape(np.repeat(x, int(1/delta_t)), (nPulses, int(1/delta_t))).transpose()

        pulse_beginnings = np.random.randint(0, seconds, int(1/delta_t))
        pulse_beginnings.sort()

        input = np.zeros((int(seconds/delta_t), 1, 2))
        target = np.zeros(int(seconds/delta_t))
        target[0:targetpulse] = np.zeros(targetpulse)
        for i in range(pulse_beginnings.shape[0]):
            input[int(pulse_beginnings[i]/delta_t):(int(pulse_beginnings[i]/delta_t)+int(1/delta_t)), 0, 0] = pulses[:, i]
            time_difference = pulse_beginnings[i] - pulse_beginnings[i-1]
            if time_difference <= 30 & time_difference >= 5:
                input[int(pulse_beginnings[i]/delta_t):(int(pulse_beginnings[i]/delta_t)+int(1/delta_t)), 0, 1] = np.ones(int(1/delta_t))
                target[int(pulse_beginnings[i]/delta_t-targetpulse):int(pulse_beginnings[i]/delta_t)] = np.repeat(np.mean((pulses[0, (i-1)], pulses[0, (i-2)])), targetpulse)

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

    mover = Avemover(100, 1000)
    input, target = mover.gen_data(100)
    model = mover.build_model((1, 2))

    model.compile(optimizer="rmsprop", loss="mse",
                  metrics=['accuracy'])

    history = model.fit(input, target, epochs=5)



