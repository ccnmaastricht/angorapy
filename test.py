from mpi4py import MPI
import tensorflow as tf
import os

mpi_comm = MPI.COMM_WORLD
gpus = tf.config.list_physical_devices('GPU')
is_gpu_process = mpi_comm.rank < len(gpus)

if not is_gpu_process:
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    tf.config.experimental.set_memory_growth(gpus[mpi_comm.rank], True)


