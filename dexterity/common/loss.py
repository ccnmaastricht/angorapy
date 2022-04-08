import tensorflow as tf


def multi_point_euclidean_distance(a: tf.Tensor, b: tf.Tensor):
    """Calculate mean euclidean distance between flattened points in 3D euclidean space.

    Args:
        a (object):     Tensor of shape (BATCH, FLATPOINTS); missing batch dimension will be inserted
        b (object):     Tensor of shape (BATCH, FLATPOINTS); missing batch dimension will be inserted
    """
    assert a.shape.as_list() == b.shape.as_list(), f"Cannot measure euclidean distance between tensors of unequal shape ({a.shape} != {b.shape})."
    assert a.shape[-1] % 3 == 0, "Flattened point dimension must be multiple of 3 to get 3D distance."

    if len(a.shape) < 2:
        a = tf.expand_dims(a, 0)
        b = tf.expand_dims(b, 0)

    batch_dim = tf.shape(a)[0]

    return tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        tf.reshape(a, (batch_dim, -1, 3)),
                        tf.reshape(b, (batch_dim, -1, 3))
                    )
                ), axis=-1
            )
        )
    )


class EuclideanDistanceLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        return multi_point_euclidean_distance(y_pred, y_true)
