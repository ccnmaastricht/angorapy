import numpy as np
import tensorflow as tf
from scipy.integrate import odeint
from scipy.special import binom


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_fit(RHS, LHS, coefficient_threshold):
    m, n = LHS.shape
    Xi = np.linalg.lstsq(RHS, LHS, rcond=None)[0]

    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:, i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds, i] = np.linalg.lstsq(RHS[:, big_inds], LHS[:, i], rcond=None)[0]
    return Xi


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
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(tf.multiply(z[:, i], z[:, j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:, i]))

    return tf.stack(library, axis=1)


def sindy_library_tf_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
    library = [tf.ones(tf.shape(z)[0])]

    z_combined = tf.concat([z, dz], 1)

    for i in range(2 * latent_dim):
        library.append(z_combined[:, i])

    if poly_order > 1:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                library.append(tf.multiply(z_combined[:, i], z_combined[:, j]))

    if poly_order > 2:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k])

    if poly_order > 3:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    for p in range(k, 2 * latent_dim):
                        library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k] * z_combined[:, p])

    if poly_order > 4:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    for p in range(k, 2 * latent_dim):
                        for q in range(p, 2 * latent_dim):
                            library.append(
                                z_combined[:, i] * z_combined[:, j] * z_combined[:, k] * z_combined[:, p] * z_combined[
                                                                                                            :, q])

    if include_sine:
        for i in range(2 * latent_dim):
            library.append(tf.sin(z_combined[:, i]))

    return tf.stack(library, axis=1)


def compute_z_derivatives(input, dx, weights, biases):
    """
    Compute the first order time derivatives by propagating through the network.
    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.
    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
    dz = dx
    for i in range(len(weights) - 1):
        input = tf.matmul(input, weights[i]) + biases[i]
        dz = tf.multiply(tf.cast(input > 0, tf.float32), tf.matmul(dz, weights[i]))
        input = tf.nn.relu(input)
    dz = tf.matmul(dz, weights[-1])

    return dz


def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    """
    Compute the first and second order time derivatives by propagating through the network.
    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.
    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
        ddz - Tensorflow array, second order time derivatives of the network output.
    """
    dz = dx
    ddz = ddx
    if activation == 'elu':
        for i in range(len(weights) - 1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz_prev = tf.matmul(dz, weights[i])
            elu_derivative = tf.minimum(tf.exp(input), 1.0)
            elu_derivative2 = tf.multiply(tf.exp(input), tf.to_float(input < 0))
            dz = tf.multiply(elu_derivative, dz_prev)
            ddz = tf.multiply(elu_derivative2, tf.square(dz_prev)) \
                  + tf.multiply(elu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    elif activation == 'relu':
        # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
        for i in range(len(weights) - 1):
            input = tf.matmul(input, weights[i]) + biases[i]
            relu_derivative = tf.to_float(input > 0)
            dz = tf.multiply(relu_derivative, tf.matmul(dz, weights[i]))
            ddz = tf.multiply(relu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights) - 1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz_prev = tf.matmul(dz, weights[i])
            sigmoid_derivative = tf.multiply(input, 1 - input)
            sigmoid_derivative2 = tf.multiply(sigmoid_derivative, 1 - 2 * input)
            dz = tf.multiply(sigmoid_derivative, dz_prev)
            ddz = tf.multiply(sigmoid_derivative2, tf.square(dz_prev)) \
                  + tf.multiply(sigmoid_derivative, tf.matmul(ddz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    else:
        for i in range(len(weights) - 1):
            dz = tf.matmul(dz, weights[i])
            ddz = tf.matmul(ddz, weights[i])
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    return dz, ddz


if __name__ == '__main__':
    print(sindy_library_tf(tf.random.normal((1024, 100), dtype=tf.float32), latent_dim=50, poly_order=2))
