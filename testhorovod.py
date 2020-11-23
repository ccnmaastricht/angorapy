import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if gpus and hvd.local_rank() < len(gpus):
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')