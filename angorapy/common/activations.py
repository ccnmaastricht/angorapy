import tensorflow as tf
from matplotlib import pyplot as plt


def _lif_roh(x, gamma=0.5):
    return gamma * tf.math.log(1 + tf.exp(x / gamma))


def lif(x, tau_rc=0.02, tau_ref=0.004, v_th=1, gamma=0.1):
    return 1 / (tau_ref + tau_rc * tf.math.log(1 + (v_th / (_lif_roh(x - v_th, gamma=gamma)))))


class LiF(tf.keras.layers.Layer):
    pass


if __name__ == '__main__':
    inputs = tf.linspace(0.7, 10, 100)

    activation = tf.keras.layers.Activation(lif)

    outputs = activation(tf.expand_dims(inputs, 0))

    plt.plot(inputs, tf.squeeze(outputs))
    plt.show()