import numpy as np
import tensorflow as tf

model = tf.keras.Sequential((
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1),
))

model(np.random.normal(size=(1, 1000)))

optimizer = tf.keras.optimizers.Adam()

for i in range(10):
    optimizer.apply_gradients(zip([np.random.normal(size=tv.shape) for tv in model.trainable_variables], model.trainable_variables))