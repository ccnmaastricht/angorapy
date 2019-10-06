#!/usr/bin/env python
"""Here I test different approaches to problems against each other in order to see which is fastest."""
import time

import numpy
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp

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


if __name__ == "__main__":
    benchmark_trajectory_filling()
