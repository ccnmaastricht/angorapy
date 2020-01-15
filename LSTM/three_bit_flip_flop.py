import tensorflow as tf
import numpy as np

class Flipflopper:


    def build_model(inputshape):
        name = "flipflopmodel"

        inputs = tf.keras.Input(shape=inputshape, name="input")

        x = tf.keras.layers.SimpleRNN(300, name='rnn')(inputs)
        #x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(3, activation="linear", name="dense")(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        model.summary()

        return model

    def generate_stim(size):

        pulse_duration = 10
        batch_size=1
        nOut = 3 # number of output channels
        nPulses = np.floor(size*1)
        input = np.zeros((size, batch_size, nOut))
        T = np.arange(0, size, int(size/nPulses))

        tstart = np.random.permutation(T)
        for i in range(int(nPulses)):
            if tstart[i] <= (size-pulse_duration):
                input[tstart[i]:tstart[i]+pulse_duration, 0, np.random.randint(0, 3)] = 2*(np.random.rand() < .5)-1

        # generate output
        target = np.zeros((size, nOut))

        prev = np.zeros((batch_size, nOut))
        for t in range(size-batch_size):
            changed = np.argwhere(input[t, :, :])
            prev[0, changed[0, 1]] = input[t, 0, changed[0, 1]]
            target[t, :] = prev


        return input, target

if __name__ == "__main__":
    size = 10000
    flopper = Flipflopper
    stim, target = flopper.generate_stim(size)

    inputshape = (1, 3)
    model = flopper.build_model(inputshape)

    model.compile(optimizer="adam", loss="mse",
                  metrics=['accuracy'])

    history = model.fit(tf.convert_to_tensor(stim, dtype=tf.float32),
                        tf.convert_to_tensor(target, dtype=tf.float32), epochs=100)

