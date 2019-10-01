#!/usr/bin/env python
import tensorflow as tf

tf.enable_eager_execution()

a = tf.distributions.Normal(tf.convert_to_tensor([[2, 2, 2], [3, 3, 3]], dtype=tf.float64), tf.convert_to_tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=tf.float64))
print(a.sample())