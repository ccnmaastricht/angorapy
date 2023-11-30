import tensorflow as tf
from tensorflow import keras


def distance_in_millimeters(a: tf.Tensor, b: tf.Tensor):
    """Compute the distance between two poses in millimeters."""
    return tf.reduce_mean(tf.norm(a[..., :3] - b[..., :3], ord="euclidean", axis=-1)) * 1000


def rotational_distance_in_degrees(a: tf.Tensor, b: tf.Tensor):
    """Compute the rotational distance between two poses in degrees."""
    batch_dot_product = tf.reduce_sum(tf.multiply(a[..., 3:], b[..., 3:]), axis=-1)
    distance = tf.acos(2 * tf.pow(batch_dot_product, 2) - 1) * 180 / 3.141592653589793

    return tf.reduce_mean(distance)
