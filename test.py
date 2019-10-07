import numpy
import ray
import tensorflow as tf


@ray.remote
class Worker:

    def __init__(self):
        self.model = tf.keras.layers.Dense(2, input_dim=5)

    def do(self, a, b):

        # c = model(a)
        return 25  #c.numpy().tolist()


ray.init()

# one, two = tf.convert_to_tensor(numpy.random.randn(5)), tf.convert_to_tensor(numpy.random.randn(5))
one, two = numpy.random.randn(5), numpy.random.randn(5)

workers = [Worker.remote() for i in range(4)]
output_ids = [w.do.remote(one, two) for w in workers]
print(output_ids)
