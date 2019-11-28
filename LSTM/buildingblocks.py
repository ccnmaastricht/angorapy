import tensorflow as tf


def _buildingblock_network(input_dim: int, output_dim: int, hidden_neurons: int,
                           batch_size: int = None, name: str = None):
    inputs = tf.keras.Input(batch_shape = (batch_size, input_dim))

    x = tf.keras.layers.Dense(hidden_neurons)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(hidden_neurons)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(output_dim, activation = "linear")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


