#!/usr/bin/env python
import time

import tensorflow as tf
import tensorflow_probability as tfp

import scipy.stats

n = 100

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
