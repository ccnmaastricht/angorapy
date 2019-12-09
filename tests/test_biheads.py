import tensorflow as tf

inp = tf.keras.Input((2,))
x = tf.keras.layers.Dense(3)(inp)
y = tf.keras.layers.Dense(4)(inp)
model_sub = tf.keras.Model(inputs=[inp], outputs=[x, y])

inp2 = tf.keras.Input((2,))
z = tf.keras.layers.Dense(3)(inp2)
sub_out = model_sub(z)
model_final = tf.keras.Model(inputs=[inp2], outputs=[z, sub_out])

model_final.predict([1, 2])
