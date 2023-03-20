from typing import List, Tuple, Iterable

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

    def apply_gradients(self, grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Variable]], name=None, **kwargs):
        """ Apply the gradients after averaging over processes."""
        if self.comm.size > 1:
            grads_and_vars = [(self.comm.allreduce(g, op=MPI.SUM) / self.comm.Get_size(), v)
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
        """Convert the Optimizer into a JSON-compatible serialization for saving."""
        config = self.get_config()
        for n, e in config.items():
            if hasattr(e, "item"):
                config[n] = e.item()

        return config

    @staticmethod
    def from_serialization(comm, serialization, var_list) -> "MpiAdam":
        if isinstance(serialization["learning_rate"], dict):
            schedule_type = serialization["learning_rate"]["class_name"]
            schedule_config = serialization["learning_rate"]["config"]
            learning_rate = getattr(tf.keras.optimizers.schedules, schedule_type).from_config(schedule_config)
        elif isinstance(serialization["learning_rate"], float):
            learning_rate = serialization["learning_rate"]
        else:
            raise ValueError("Unknown learning rate/schedule format.")

        adam = MpiAdam(
            comm=comm,
            learning_rate=learning_rate,
            epsilon=serialization["epsilon"]
        )

        # adam._create_all_weights(var_list)
        adam.apply_gradients(zip([tf.zeros_like(v) for v in var_list], var_list))
        adam.set_weights([tf.convert_to_tensor(v) for v in serialization["weights"][:len(adam.variables())]])  # todo remove the slicing hack; only for backwards compatibility

        return adam
