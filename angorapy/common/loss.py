import tensorflow as tf
import tensorflow_graphics as tfg


@tf.function
def euclidean_distance(a: tf.Tensor, b: tf.Tensor):
    return tf.reduce_mean(tf.norm(a[:, :3] - b[:, :3], ord="euclidean", axis=-1))


# GEODESIC
@tf.function
def geodesic_loss(a: tf.Tensor, b: tf.Tensor):
    batch_dot_product = tf.reduce_sum(tf.multiply(a[:, 3:], b[:, 3:]), axis=-1)
    distance = tf.reduce_mean(1.0 - tf.pow(batch_dot_product, 2))

    return distance


@tf.function
def geodist(a: tf.Tensor, b: tf.Tensor):
    batch_dot_product = tf.reduce_sum(tf.multiply(a[:, 3:], b[:, 3:]), axis=-1)
    distance = tf.acos(2 * batch_dot_product)

    return distance


class PoseEstimationLoss(tf.keras.losses.Loss):

    beta = 0.001

    @tf.function
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # find the euclidean distance between the two positions in the pose
        position_loss = euclidean_distance(y_pred, y_true)

        # find the shortest geodesic rotation between the two quaternions in the pose
        rotational_loss = geodesic_loss(y_pred, y_true)

        return position_loss + rotational_loss
