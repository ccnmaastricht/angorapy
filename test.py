#!/usr/bin/env python
"""Here I test different approaches to problems against each other in order to see which is fastest."""
import shutil
import time

import gym
import numpy
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp

from models.fully_connected import PPOActorFNN

n = 1000


def benchmark_sampling():
    start = time.time()
    for _ in range(n):
        distribution = tfp.distributions.Normal(3, 1)
        sample = distribution.sample()
        prob = distribution.prob(sample)
    print(f"Distribution: {round(time.time() - start, 2)}")

    start = time.time()
    for _ in range(n):
        sample = tf.random.normal([1], tf.cast([3], dtype=tf.float64), 1)
        prob = scipy.stats.norm.pdf(sample, loc=3, scale=1)
    print(f"Function: {round(time.time() - start, 2)}")


def benchmark_trajectory_filling():
    list_length = 50000

    start = time.time()
    for _ in range(n):
        a = numpy.ndarray()
        for i in range(list_length):
            a.append([2, 3, 4, 5, 6, 7, 8, 9])
    print(f"Numpy: {round(time.time() - start, 2)}")

    start = time.time()
    for _ in range(n):
        a = []
        # for i in range(list_length):
        #     a.append(numpy.array([2, 3, 4, 5, 6, 7, 8, 9]))
        # a = numpy.array(a)
    print(f"List: {round(time.time() - start, 2)}")


def benchmark_network_propagation():
    n = 10

    start = time.time()
    for _ in range(n):
        model = PPOActorFNN(gym.make("CartPole-v1"))
        model.save("saved_models/test")
        new_model = tf.keras.models.load_model("saved_models/test")
        shutil.rmtree("saved_models/test")
    print(f"Saving: {round(time.time() - start, 2)}")

    start = time.time()
    for _ in range(n):
        model = PPOActorFNN(gym.make("CartPole-v1"))
        weights = model.get_weights()
        new_model = PPOActorFNN(gym.make("CartPole-v1"))
        new_model.set_weights(weights)
    print(f"Serializing: {round(time.time() - start, 2)}")


if __name__ == "__main__":
    benchmark_network_propagation()
