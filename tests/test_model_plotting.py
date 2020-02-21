import tensorflow as tf

model = tf.keras.Sequential((
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(20),
    tf.keras.layers.Dense(30),)
)

tf.keras.utils.plot_model(model, to_file="model.svg")
tf.keras.utils.model_to_dot(model)
