from typing import List, Tuple

import tensorflow as tf
from mpi4py import MPI


class AdamSynchronizationError(Exception):
    pass


class MpiAdam(tf.keras.optimizers.Adam):
    """Adam optimizer that averages gradients across mpi processes."""

    def __init__(self, comm, learning_rate=0.001, epsilon=1e-7):
        super().__init__(learning_rate=learning_rate, epsilon=epsilon)
        self.comm = comm

        ranks = self.comm.gather(MPI.COMM_WORLD.rank)
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"An MPI Optimizer with {self.comm.size} ranks has been created; the following ranks optimize: {ranks}")

    def _sync_parameters(self):
        root_variables = self.comm.bcast(self.weights(), int_root=0)
        self.set_weights(root_variables)

    def apply_gradients(self, grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]], name=None, **kwargs):
        """ Apply the gradients after averaging over processes."""
        if self.comm.size > 1:
            grads_and_vars = [(tf.divide(self.comm.allreduce(g, op=MPI.SUM), self.comm.Get_size()), v)
                              for g, v in grads_and_vars]

        context = super().apply_gradients(grads_and_vars)

        # wait for all processes before continueing
        self.comm.Barrier()
        return context

    def validate_synced_parameters(self):
        """Validate if all optimization processes share the same Adam parameters (moments)."""
        all_variables = self.comm.gather(self.weights())
        is_in_sync = all(elem == all_variables[0] for elem in all_variables)

        if not is_in_sync:
            raise AdamSynchronizationError("Parameters of MPIAdam optimizers are out of sync.")

    def serialize(self):
        weights = self.get_weights()
        if len(weights) > 0:
            weights[0] = weights[0].item()
            weights[1:] = [w.tolist() for w in weights[1:]]

        config = self.get_config()
        for n, e in config.items():
            if hasattr(e, "item"):
                config[n] = e.item()

        return {
            **config,
            "weights": weights
        }

    @staticmethod
    def from_serialization(comm, serialization) -> "MpiAdam":
        adam = MpiAdam(
            comm=comm,
            learning_rate=serialization["learning_rate"],
            epsilon=serialization["epsilon"]
        )

        adam.set_weights(serialization["weights"])

        return adam
