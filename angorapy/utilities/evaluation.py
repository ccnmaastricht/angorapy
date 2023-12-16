import json
import statistics
import time

import statsmodels.stats.api as sms
from angorapy.utilities.datatypes import mpi_condense_stats

from angorapy.agent.ppo_agent import PPOAgent

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def evaluate_agent(agent: PPOAgent, n_episodes: int, act_confidently=False):
    """Evaluate an agent on its environment.

    Args:
        agent (Agent): The agent to evaluate.
        n_episodes (int): The number of episodes to evaluate the agent on.
        act_confidently (bool): If True, act deterministically without stochasticity.

    Returns:
        (list, list): A tuple of lists containing the episode rewards and episode lengths.
    """

    if MPI is not None:
        mpi_comm = MPI.COMM_WORLD
        rank = mpi_comm.rank
        is_root = rank == 0
        comm_size = mpi_comm.size
    else:
        is_root = True
        rank = 0
        comm_size = 1

    start = time.time()

    if is_root:
        print(
            f"Agent {agent} successfully loaded from state 'best' (training performance: {agent.cycle_reward_history[-1]}).")

    # determine number of repetitions on this worker
    worker_base, worker_extra = divmod(n_episodes, comm_size)
    worker_split = [worker_base + (r < worker_extra) for r in range(comm_size)]
    workers_n = worker_split[rank]

    stat_bundles, _ = agent.evaluate(workers_n, act_confidently=act_confidently)
    stats = mpi_condense_stats([stat_bundles])

    if is_root:
        average_reward = round(statistics.mean(stats.episode_rewards), 2)
        average_length = round(statistics.mean(stats.episode_lengths), 2)
        std_reward = round(statistics.stdev(stats.episode_rewards), 2)
        std_length = round(statistics.stdev(stats.episode_lengths), 2)

        print(
            f"Evaluated agent on {stats.numb_completed_episodes} x {agent.env_name} and achieved an average reward of {average_reward} [std: {std_reward}; "
            f"between ({min(stats.episode_rewards)}, {max(stats.episode_rewards)})].\n ")

        print("Auxiliary Performance Measures:")
        for aux_perf_name, aux_perf_trace in stats.auxiliary_performances.items():
            aux_perf_stats = sms.DescrStatsW(aux_perf_trace)
            average_perf = round(aux_perf_stats.mean, 2)
            confidence_interval = round(aux_perf_stats.tconfint_mean()[0], 2), round(aux_perf_stats.tconfint_mean()[1],
                                                                                     2)
            print(f"\t{aux_perf_name}: {average_perf} [{confidence_interval}]")
        print("\n")

        print(f"The agent {'acted confidently' if act_confidently else 'acted stochastically.'}\n"
              f"An episode on average took {average_length} steps [std: {std_length}; "
              f"between ({min(stats.episode_lengths)}, {max(stats.episode_lengths)})].\n"
              f"This took me {round(time.time() - start, 2)}s.")

        return stats
