import os
import time

import gym
from mpi4py import MPI

from agent.gather import Gatherer
from agent.policies import GaussianPolicyDistribution
from models import build_ffn_models
from utilities.util import env_extract_dims
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper

if __name__ == "__main__":
    """Performance Measuring."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env_n = "HalfCheetah-v2"
    environment = gym.make(env_n)
    distro = GaussianPolicyDistribution(environment)
    builder = build_ffn_models
    sd, ad = env_extract_dims(environment)
    wrapper = CombiWrapper((StateNormalizationWrapper(sd), RewardNormalizationWrapper()))

    n_actors = 8
    base, extra = divmod(n_actors, size)
    n_actors_on_this_node = base + (rank < extra)

    t = time.time()
    actors = [Gatherer(builder.__name__, distro.__class__.__name__, env_n, i) for i in range(n_actors_on_this_node)]

    it = time.time()
    outs_ffn = [actor.collect(512, 0.99, 0.95, 16, wrapper.serialize()) for actor in actors]
    gathering_msg = f"Gathering Time: {time.time() - it}"

    msgs = comm.gather(gathering_msg, root=0)

    if rank == 0:
        for msg in msgs:
            print(msg)

        print(f"Program Runtime: {time.time() - t}")

    MPI.Finalize()

    # remote function, 8 workers, 2048 horizon: Program Runtime: 24.98351287841797
    # remote function, 1 worker, 2048 horizon: Program Runtime: 10.563997030258179
