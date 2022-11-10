import tensorflow as tf
from matplotlib import pyplot as plt


def _lif_roh(x, gamma=0.5):
    return gamma * tf.math.log(1 + tf.exp(x / gamma))


def lif(x, tau_rc=0.02, tau_ref=0.004, v_th=1, gamma=0.1):
    return 1 / (tau_ref + tau_rc * tf.math.log(1 + (v_th / (_lif_roh(x - v_th, gamma=gamma)))))


class LiF(tf.keras.layers.Layer):

    def __init__(self, tau_rc=0.02, tau_ref=0.004, v_th=1, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.v_th = v_th
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc
        self.gamma = gamma

    def call(self, inputs, **kwargs):
        return lif(inputs, self.tau_rc, self.tau_ref, self.v_th, self.gamma)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "v_th": self.v_th,
            "tau_ref": self.tau_ref,
            "tau_rc": self.tau_rc,
            "gamma": self.gamma
        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    inputs = tf.linspace(0.7, 10, 100)

    activation = LiF()
    print(activation.get_config())

    outputs = activation(tf.expand_dims(inputs, 0))

    plt.plot(inputs, tf.squeeze(outputs))
    plt.show()