import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import odeint
from scipy.io import loadmat
from analysis.sindy.components import build_autoencoder
import tensorflow as tf
import pysindy as ps


# Generate training data
def lorenz(x, t):
    return [
        10 * (x[1] - x[0]),
        x[0] * (28 - x[2]) - x[1],
        x[0] * x[1] - 8 / 3 * x[2],
    ]


def train(opt, sindy_autoencoder, x, dx):
    with tf.GradientTape() as tape:
        x_hat, z, dz, dx_decode, sindy_predict = sindy_autoencoder([x, dx])
        loss = 0.1* tf.reduce_mean((x-x_hat)**2) \
        + 0.2* tf.reduce_mean((dz - sindy_predict)**2) \
        + 0.2* tf.reduce_mean((dx - dx_decode)**2)
        gradients = tape.gradient(loss, sindy_autoencoder.trainable_variables)
        gradient_variables = zip(gradients, sindy_autoencoder.trainable_variables)
        opt.apply_gradients(gradient_variables)
        print(loss.numpy())


if __name__ == "__main__":
    # Fit the models and simulate

    poly_order = 5
    threshold = 0.05
    seed = 100
    np.random.seed(seed)  # Seed for reproducibility
    dt = 0.001
    noise_level = 1e-3

    t_sim = np.arange(0, 20, dt)
    t_train = np.arange(0, 100, dt)
    x0_train = [-8, 8, 27]
    x_train = odeint(lorenz, x0_train, t_train)
    print(x_train.shape)
    x_dot_train_measured = np.array(
        [lorenz(x_train[i], 0) for i in range(t_train.size)]
    )

    batch_size = 10
    sindy_autoencoder = build_autoencoder(input_dim=3, hidden_dim=3, z_dim=3, n_hidden=2,
                                          poly_order=2, batch_size=batch_size)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    for i in range(int(len(x_train)/batch_size-1)):
        x = x_train[i:i+batch_size, :]
        dx = x_dot_train_measured[i:i+batch_size, :]
        train(opt, sindy_autoencoder, x, dx)
