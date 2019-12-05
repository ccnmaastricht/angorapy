import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# NOT NESTED
inp = tf.keras.Input((4,))
y = tf.keras.layers.Dense(4, name="od_1")(inp)
y = tf.keras.layers.Dense(2, name="od_2")(y)
y = tf.keras.layers.Dense(4, name="id_1")(y)
y = tf.keras.layers.Dense(10, name="od_3")(y)
y = tf.keras.layers.Dense(10, name="od_4")(y)
final_model = tf.keras.Model(inputs=[inp], outputs=[y])
final_model.summary()

sub_model = tf.keras.Model(inputs=[final_model.input], outputs=[final_model.get_layer("id_1").output])
sub_model.summary()

# NESTED

inp_1 = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(4, name="id_1")(inp_1)
inner_model = tf.keras.Model(inputs=[inp_1], outputs=[x], name="inner_model")

inp_outer = tf.keras.Input((4,))
y = tf.keras.layers.Dense(4, name="od_1")(inp_outer)
y = tf.keras.layers.Dense(2, name="od_2")(y)
y = inner_model(y)
y = tf.keras.layers.Dense(10, name="od_3")(y)
y = tf.keras.layers.Dense(10, name="od_4")(y)
final_model = tf.keras.Model(inputs=[inp_outer], outputs=[y])
final_model.summary()

sub_model = tf.keras.Model(inputs=[final_model.input], outputs=[final_model.get_layer("inner_model").get_layer("id_1").output])
sub_model.summary()

