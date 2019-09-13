#!/usr/bin/env python
"""TODO Module Docstring."""
import numpy
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

with tf.device("GPU:0"):
  x = [[1, 2],
      [2, 3],
      [4, 5]]

  y = [
      [10, 20],
      [20, 30],
      [40, 50]
  ]

  z = numpy.array([
      [15, 25],
      [25, 35],
      [45, 55]
  ])

  data = tf.data.Dataset.from_tensor_slices({
      "state": x,
      "action": y,
      "reward": z
  })

  for p in data:
      print(p)