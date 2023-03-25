#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""

import argparse
import json
import os
import sys
import statsmodels.stats.api as sms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from angorapy.environments.hand.shadowhand import BaseShadowHandEnv
from utilities.datatypes import mpi_condense_stats
import statistics
import time

from agent.ppo_agent import PPOAgent
from common.const import BASE_SAVE_PATH
from mpi4py import MPI
import tensorflow as tf

tf.get_logger().setLevel('INFO')

parser = argparse.ArgumentParser(description="Evaluate an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=None)
parser.add_argument("-n", type=int, help="number of evaluation episodes", default=12)
parser.add_argument("--act-confidently", action="store_true", help="act deterministically without stochasticity")
args = parser.parse_args()

mpi_comm = MPI.COMM_WORLD
is_root = mpi_comm.rank == 0

if args.id is None:
    ids = map(int, os.listdir(BASE_SAVE_PATH))
    args.id = max(ids)

start = time.time()
agent = PPOAgent.from_agent_state(args.id, "best", path_modifier="../")

# if isinstance(agent.env.unwrapped, BaseShadowHandEnv):
#     agent.env.env.set_delta_t_simulation(0.0005)

if is_root:
    print(f"Agent {args.id} successfully loaded from state 'best' (training performance: {agent.cycle_reward_history[-1]}).")


# determine number of repetitions on this worker
worker_base, worker_extra = divmod(args.n, MPI.COMM_WORLD.size)
worker_split = [worker_base + (r < worker_extra) for r in range(MPI.COMM_WORLD.size)]
workers_n = worker_split[mpi_comm.rank]

stat_bundles, _ = agent.evaluate(workers_n, act_confidently=args.act_confidently)
stats = mpi_condense_stats([stat_bundles])

if is_root:
    average_reward = round(statistics.mean(stats.episode_rewards), 2)
    average_length = round(statistics.mean(stats.episode_lengths), 2)
    std_reward = round(statistics.stdev(stats.episode_rewards), 2)
    std_length = round(statistics.stdev(stats.episode_lengths), 2)

    print(f"Evaluated agent on {stats.numb_completed_episodes} x {agent.env_name} and achieved an average reward of {average_reward} [std: {std_reward}; "
          f"between ({min(stats.episode_rewards)}, {max(stats.episode_rewards)})].\n ")

    print("Auxiliary Performance Measures:")
    for aux_perf_name, aux_perf_trace in stats.auxiliary_performances.items():
        aux_perf_stats = sms.DescrStatsW(aux_perf_trace)
        average_perf = round(aux_perf_stats.mean, 2)
        confidence_interval = round(aux_perf_stats.tconfint_mean()[0], 2), round(aux_perf_stats.tconfint_mean()[1], 2)
        print(f"\t{aux_perf_name}: {average_perf} [{confidence_interval}]")
    print("\n")

    print(f"The agent {'acted confidently' if args.act_confidently else 'acted stochastically.'}\n"
          f"An episode on average took {average_length} steps [std: {std_length}; "
          f"between ({min(stats.episode_lengths)}, {max(stats.episode_lengths)})].\n"
          f"This took me {round(time.time() - start, 2)}s.")

    with open(f"../{agent.experiment_directory}/evaluation.json", "w") as f:
        json.dump(stats._asdict(), f)
