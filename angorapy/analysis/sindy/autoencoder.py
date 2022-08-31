from typing import Iterable

import numpy as np
import tensorflow as tf

from angorapy.analysis.util.sindy import compute_z_derivatives, sindy_library_tf, library_size


def build_encoder(layer_sizes: Iterable[int]):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation="relu") for size in layer_sizes
    ])

    return encoder


def build_decoder(layer_sizes: Iterable[int]):
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation="relu" if i != (len(layer_sizes) - 1) else None) for i, size in
        enumerate(layer_sizes)
    ])

    return decoder


def build_full_autoencoder(layer_sizes: Iterable[int], original_dim: int):
    input_layer = tf.keras.layers.Input((original_dim,))

    encoded = build_encoder(layer_sizes)(input_layer)
    decoded = build_decoder(layer_sizes[1:] + [original_dim])(encoded)

    model = tf.keras.Model(inputs=[input_layer], outputs=[encoded, decoded])

    return model


class SindyAutoencoder(tf.keras.Model):

    def __init__(self, layer_sizes: Iterable[int], z_dim, original_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.z_dim = z_dim

        # autoencoder
        self.encoder = build_encoder(layer_sizes)
        self.decoder = build_decoder(layer_sizes[1:] + [original_dim])

        # sindy
        self.coefficients = tf.Variable(
            initial_value=tf.random_normal_initializer()(shape=(library_size(self.z_dim, 2), self.z_dim)),
            trainable=True
        )

    def call(self, inputs, training=None, mask=None):
        z_coordinates = self.encoder(inputs)
        x_reconstruction = self.decoder(z_coordinates)

        z_library = sindy_library_tf(z_coordinates, self.z_dim, poly_order=2)
        z_derivative_approximation = tf.matmul(z_library, self.coefficients)

        return z_coordinates, x_reconstruction, z_derivative_approximation
