import autograd.numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_layer(params, x):
    return sigmoid(np.matmul(x, params[0]) + params[1])


def build_encoder(layer_sizes, scale=1e-2):
    encoding_params = []
    for m, n in zip(layer_sizes[1:], layer_sizes[:-1]):
        w, b = scale * np.random.randn(n, m), scale * np.random.randn(m)
        encoding_params.append([w, b])
    return encoding_params


def build_decoder(layer_sizes, scale=1e-2):
    decoding_params = []
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):

        w, b = scale * np.random.randn(n, m), scale * np.random.randn(m)
        decoding_params.append([w, b])
    return decoding_params


def z_derivative(params, x, dx):
    dz = dx

    for w, b in params[:-1]:
        x = np.matmul(x, w) + b
        x = sigmoid(x)
        dz = np.multiply(np.multiply(x, 1-x), np.matmul(dz, w))

    return np.matmul(dz, params[-1][0])


def z_derivative_decode(params, z, sindy_predict):
    dx_decode = sindy_predict
    params = params[::-1]

    for w, b in params[:-1]:
        z = np.matmul(z, w) + b
        z = sigmoid(z)
        dx_decode = np.multiply(np.multiply(z, 1-z), np.matmul(dx_decode, w))

    return np.matmul(dx_decode, params[-1][0])


def encoding_pass(params, x):
    activation = x

    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return np.matmul(activation, params[-1][0]) + params[-1][1]


def decoding_pass(params, input):
    activation = input
    params = params[::-1]  # reverse order for decoder
    for w, b in params[:-1]:
        activation = sigmoid_layer([w, b], activation)

    return np.matmul(activation, params[-1][0]) + params[-1][1]

