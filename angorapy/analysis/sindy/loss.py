import tensorflow as tf


def sindy_ae_reconstruction_loss(inputs, reconstruction):
    return tf.keras.losses.mean_squared_error(inputs, reconstruction)


def sindy_ae_loss_in_x(dx, z_coordinates, ):
    pass


def sindy_ae_loss_in_z():
    pass


def sindy_regularization():
    pass
