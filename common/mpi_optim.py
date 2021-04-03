from typing import List, Tuple
from mpi4py import MPI

import tensorflow as tf


class AdamSynchronizationError(Exception):
    pass


class MpiAdam(tf.keras.optimizers.Adam):
    """Adam optimizer that averages gradients across mpi processes."""

    def __init__(self, comm, learning_rate=0.001, epsilon=1e-7):
        super().__init__(learning_rate=learning_rate, epsilon=epsilon)
        self.comm = comm

    def _sync_parameters(self):
        root_variables = self.comm.bcast(self.weights(), int_root=0)
        self.set_weights(root_variables)

    def apply_gradients(self, grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]], name=None, **kwargs):
        """ Apply the gradients after averaging over processes."""
        reduced_grads_and_vars = [(tf.divide(self.comm.allreduce(g, op=MPI.SUM), self.comm.Get_size()), v)
                                  for g, v in grads_and_vars]
        super().apply_gradients(reduced_grads_and_vars)

        # wait for all processes before continueing
        self.comm.Barrier()

    def validate_synced_parameters(self):
        """Validate if all optimization processes share the same Adam parameters (moments)."""
        all_variables = self.comm.gather(self.weights())
        is_in_sync = all(elem == all_variables[0] for elem in all_variables)

        if not is_in_sync:
            raise AdamSynchronizationError("Parameters of MPIAdam optimizers are out of sync.")