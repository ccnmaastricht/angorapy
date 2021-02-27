import numpy
import tensorflow as tf

from common.senses import Sensation

vision = tf.keras.Input(batch_shape=(None, 3), name="vision")
touch = tf.keras.Input(batch_shape=(None, 4), name="somatosensation")

conc = tf.keras.layers.Concatenate()([vision, touch])
dense = tf.keras.layers.Dense(1)(conc)

model = tf.keras.Model(inputs={"vision": vision, "somatosensation": touch}, outputs={"out": dense})

sens = Sensation(vision=numpy.random.normal(size=(3,)), somatosensation=numpy.random.normal(size=(4,)))

sens.with_leading_dims()
print(sens.dict())
print(model.predict(sens.dict()))