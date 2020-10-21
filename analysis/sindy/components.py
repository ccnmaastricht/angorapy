import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import concatenate, Dense, ReLU


def build_autoencoder(input_dim: int, hidden_dim: int, z_dim: int, n_hidden: int,
                      poly_order:int = None, batch_size: int = None):
    # encoder
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, input_dim), name="x")
    dx = tf.keras.layers.Input(batch_shape=(batch_size, input_dim), name="dx")
    dz = dx
    x = inputs
    for i in range(n_hidden):
        shared_x = Dense(hidden_dim, name="shared_encoder"+str(i+1))
        x = shared_x(x)
        weights = shared_x.get_weights()[0]
        dz = tf.multiply(tf.cast(x > 0, tf.float32), tf.matmul(dz, weights))
        x = ReLU()(x)

    shared_x = Dense(z_dim, name="z")
    z = shared_x(x) # last layer that produces latent space does not have an activation
    weights = shared_x.get_weights()[0]
    dz = tf.matmul(dz, weights, name="dz")

    Theta = concatenate([tf.ones(tf.shape(z)), z])
    if poly_order > 1:
        poly_order_two = []
        for i in range(z_dim):
            for j in range(i, z_dim):
                poly_order_two.append(tf.multiply(tf.slice(z, [0, i], [batch_size, 1]),
                                                  tf.slice(z, [0, j], [batch_size, 1])))
        Theta = concatenate([tf.ones(tf.shape(z)), z,
                             tf.squeeze(tf.stack(poly_order_two, axis=1))])

    w_init = tf.random_normal_initializer()
    sindy_coefficients = tf.Variable(initial_value=w_init((Theta.shape[1], z_dim)), trainable=True, name="SINDy_coefficients")

    sindy_predict = tf.matmul(Theta, sindy_coefficients, name="SINDy")
    x, dx_decode = z, sindy_predict

    # decoder
    for i in range(n_hidden):
        shared_x_decode = tf.keras.layers.Dense(hidden_dim, name="shared_decoder"+str(i+1))
        x = shared_x_decode(x)
        weights = shared_x_decode.get_weights()[0]
        dx_decode = tf.multiply(tf.cast(x > 0, tf.float32), tf.matmul(dx_decode, weights))
        x = tf.keras.layers.ReLU()(x)

    shared_x_decode = tf.keras.layers.Dense(input_dim, name="x_hat")
    x_hat = shared_x_decode(x)
    weights = shared_x_decode.get_weights()[0]
    dx_decode = tf.matmul(dx_decode, weights, name="dx_decode")

    return tf.keras.Model(inputs=[inputs, dx], outputs=[x_hat, z, dz, dx_decode, sindy_predict], name="Autoencoder")


if __name__ == "__main__":
    autoencoder = build_autoencoder(input_dim=10, hidden_dim=5,
                                    z_dim=3, n_hidden=1, poly_order=2, batch_size=4)
    autoencoder.summary()

    x = tf.random.uniform((4, 10))
    dx = tf.random.uniform((4, 10))

    x_hat, z, dz, dx_decode, sindy_predict = autoencoder([x, dx])
    tf.keras.utils.plot_model(autoencoder, to_file="model.png",
                              show_shapes=True, dpi=300)

