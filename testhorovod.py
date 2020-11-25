from mpi4py import MPI
import horovod.tensorflow as hvd

# Split COMM_WORLD into subcommunicators
subcomm = MPI.COMM_WORLD.Split(color=0 if MPI.COMM_WORLD.rank < 2 else 1,
                               key=MPI.COMM_WORLD.rank)

# Initialize Horovod
hvd.init(comm=subcomm)

if MPI.COMM_WORLD.rank < 2:
    print(f'COMM_WORLD rank: {MPI.COMM_WORLD.rank}|{MPI.COMM_WORLD.size}, '
          f'Horovod rank: {hvd.rank()}|{hvd.size()}, '
          f'Subcomm rank: {subcomm.rank}|{subcomm.size}')
