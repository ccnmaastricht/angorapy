import tensorflow as tf
import numpy as np


def build_autoencoder(input_dim: int, hidden_dim: int, z_dim: int, n_hidden: int,
                      batch_size: int = None):
    # encoder
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, input_dim), name="x")
    dx = tf.keras.layers.Input(batch_shape=(batch_size, input_dim), name="dx")
    dz = dx
    x = inputs
    for i in range(n_hidden):
        shared_x = tf.keras.layers.Dense(hidden_dim, name="shared_encoder")
        x = shared_x(x)
        weights = shared_x.get_weights()[0]
        dz = tf.multiply(tf.cast(x > 0, tf.float32), tf.matmul(dz, weights))
        x = tf.keras.layers.ReLU()(x)

    shared_x = tf.keras.layers.Dense(z_dim)
    z = shared_x(x)
    weights = shared_x.get_weights()[0]
    dz = tf.matmul(dz, weights)


    Theta = sindy_library_tf(z, hidden_dim, poly_order=1)
    Theta = tf.squeeze(Theta, axis=-1)
    print(Theta.shape)
    w_init = tf.random_normal_initializer()
    sindy_coefficients = tf.Variable(initial_value=w_init((6, z_dim)), trainable=True)
    print(sindy_coefficients.shape)
    sindy_predict = tf.matmul(Theta, sindy_coefficients)

    x, dx_decode = z, sindy_predict
    # decoder
    for i in range(n_hidden):
        shared_x_decode = tf.keras.layers.Dense(hidden_dim, name="shared_decoder")
        x = shared_x_decode(x)
        weights = shared_x_decode.get_weights()[0]
        dx_decode = tf.multiply(tf.cast(x > 0, tf.float32), tf.matmul(dx_decode, weights))
        x = tf.keras.layers.ReLU()(x)

    shared_x_decode = tf.keras.layers.Dense(input_dim, name="x_hat")
    x_hat = shared_x_decode(x)
    weights = shared_x_decode.get_weights()[0]
    dx_decode = tf.matmul(dx_decode, weights)

    return tf.keras.Model(inputs=[inputs, dx], outputs=[x_hat, dz, dx_decode, z, sindy_predict], name="Autoencoder")


def sindy_library_tf(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.
    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.
    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [tf.ones((tf.shape(z)[0], 1))]

    for i in range(latent_dim):
        library.append(tf.slice(z, [0, i], [4, 1]))

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(tf.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:,i]))

    return tf.stack(library, axis=1)


if __name__ == "__main__":
    autoencoder = build_autoencoder(input_dim=10, hidden_dim=5, z_dim=3, n_hidden=1, batch_size=4)
    autoencoder.summary()

    x = tf.random.uniform((4, 10))
    dx = tf.random.uniform((4, 10))

    x_hat, dz, dx_decode, z, sindy_predict = autoencoder([x, dx])
