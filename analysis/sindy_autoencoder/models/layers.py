import jax.numpy as jnp


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def relu(x):
    return jnp.max((0, x))


def sigmoid_layer(params, x):
    return sigmoid(jnp.matmul(params[0], x) + params[1])


def relu_layer(params, x):
    return relu(jnp.matmul(params[0], x) + params[1])
