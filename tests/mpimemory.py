import os

import psutil

from mpi4py import MPI
import numpy as np

import tensorflow as tf
from tqdm import tqdm


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
model = tf.keras.Sequential((
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(20),
    tf.keras.layers.Dense(30),
    tf.keras.layers.Dense(40),
))

model(np.random.normal(size=(1, 5)))

optimizer = tf.keras.optimizers.Adam()


def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


with tqdm() as pbar:
    for _ in range(1000):
        if rank == 0:
            print(f"rank {rank}, memory usage = {get_memory_usage():.3f} Mo")
        for _ in range(1000):
            # case 0: allreduce
            # memory leak
            optimizer.apply_gradients(
                zip([tf.divide(comm.allreduce(np.random.normal(size=tv.shape), op=MPI.SUM), tf.cast(comm.Get_size(), dtype=tf.float32)) for tv in model.trainable_variables],
                    model.trainable_variables)
            )

            # case 1: reduce
            # no memory leak
            # result = comm.reduce(hist, op=MPI.SUM, root=0)
            # case 2: Allreduce
            # no memory leak
            # result = np.empty_like(hist)
            # comm.Allreduce(hist, result, op=MPI.SUM)

            pbar.update(1)

        # assert result[0] == 0
        # assert result[1] == comm.size