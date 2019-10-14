#!/usr/bin/env python
"""TODO Module Docstring."""

import numpy as np
import ray
from tensorflow.keras import layers


def create_keras_model():
    import tensorflow as tf

    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation="relu", input_shape=(32,)))
    # Add another:
    model.add(layers.Dense(64, activation="relu"))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.train.RMSPropOptimizer(0.01),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


ray.init()


@ray.remote
class Network(object):
    def __init__(self):
        self.model = create_keras_model()
        self.dataset = np.random.random((1000, 32))
        self.labels = random_one_hot_labels((1000, 10))

    def train(self):
        history = self.model.fit(self.dataset, self.labels, verbose=False)
        return history.history

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        # Note that for simplicity this does not handle the optimizer state.
        self.model.set_weights(weights)


NetworkActor = Network.remote()
result_object_id = NetworkActor.train.remote()
ray.get(result_object_id)
