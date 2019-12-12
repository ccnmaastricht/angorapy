import os

import tensorflow as tf

from utilities.util import set_all_seeds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

set_all_seeds(1)

inp = tf.keras.Input(shape=(None, 1))
x = tf.keras.layers.Masking(mask_value=0)(inp)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))(inp)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4))(x)
x = tf.keras.layers.SimpleRNN(3, return_sequences=True)(x)
x = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inp, outputs=x)

print(f"\n----TD1---")
print(model.get_layer("time_distributed_1").weights)
print("---------------\n")

a = tf.expand_dims(
    tf.keras.preprocessing.sequence.pad_sequences([[1, 2, 3, 4],
                                                   [1, 2, 3],
                                                   [2, 3],
                                                   [3, 4, 5, 6, 7, 7],
                                                   [3]],
                                                  padding="post", dtype="float32", maxlen=7), axis=-1)

with tf.GradientTape() as tape:
    out = model(a)
    loss = out - 10

gradients = tape.gradient(loss, model.trainable_variables)
for i, grad in enumerate(gradients):
    print(model.trainable_variables[i].name)
    print(grad)
    print("\n")


print("\n\n")
print(out)
