#!/usr/bin/env python
"""Read and Write tensorflow datasets."""
import numpy
import tensorflow as tf


def read_tf_dataset(dataset: tf.data.Dataset, filename: str):
    pass


def write_tf_dataset(dataset: tf.data.Dataset, filename: str):
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)


if __name__ == "__main__":
    data = numpy.random.randint(0, 5, 10000)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    fname = "test.tfrecord"

    write_tf_dataset(dataset, fname)

