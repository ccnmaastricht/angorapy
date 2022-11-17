import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K


# @tf.function
def lif(
    x, tau_rc=0.02, tau_ref=0.004, v_th=1., gamma=0.01, validity_threshold=-20., js_threshold=30.
):
    js = x / gamma
    j_valid = js > validity_threshold

    # splitting tf.where in two steps to avoid gradient nans
    safe_js_z = tf.where(
        js < js_threshold,
        js,
        tf.ones_like(js)
    )

    z = tf.where(
        js < js_threshold,
        tf.math.log1p(tf.math.exp(safe_js_z)),
        js,
    ) * gamma

    # splitting tf.where in two steps to avoid gradient nans
    safe_zs = tf.where(
        j_valid,
        z,
        tf.ones_like(z)
    )

    q = tf.where(
        j_valid,
        tf.math.log1p(1. / safe_zs),
        -js - tf.math.log(gamma)
    )

    response = v_th / (tau_ref + tau_rc * q)

    return tf.cast(response, x.dtype)


class LiF(tf.keras.layers.Layer):

    def __init__(self, tau_rc=0.02, tau_ref=0.004, v_th=1, gamma=0.01, **kwargs):
        super().__init__(**kwargs)
        self.v_th = tf.cast(v_th, dtype=tf.float32)
        self.tau_ref = tf.cast(tau_ref, dtype=tf.float32)
        self.tau_rc = tf.cast(tau_rc, dtype=tf.float32)
        self.gamma = tf.cast(gamma, dtype=tf.float32)

        self.js_threshold = tf.cast(30., dtype=tf.float32)
        self.validity_threshold = tf.cast(-20., dtype=tf.float32)

        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return lif(
            inputs,
            gamma=self.gamma,
            v_th=self.v_th,
            tau_ref=self.tau_ref,
            tau_rc=self.tau_rc,
            js_threshold=self.js_threshold,
            validity_threshold=self.validity_threshold,
        )

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
    inputs = tf.linspace(-2, 10, 100)

    for i in range(10):
        activation = LiF(tau_rc=0.02 / (i + 1))
        outputs = activation(tf.expand_dims(inputs, 0))

        plt.plot(inputs, tf.squeeze(outputs))

    plt.show()
