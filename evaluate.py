#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""
import argparse
import os
import statistics
import time

from agent.ppo import PPOAgent
from utilities.const import BASE_SAVE_PATH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="Evaluate an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=None)
parser.add_argument("-n", type=int, help="number of evaluation episodes", default=10)
args = parser.parse_args()

if args.id is None:
    ids = map(int, os.listdir(BASE_SAVE_PATH))
    args.id = max(ids)

start = time.time()
agent = PPOAgent.from_agent_state(args.id)
print(f"Agent {args.id} successfully loaded.")

lengths, rewards = agent.evaluate(args.n)

average_reward = round(statistics.mean(rewards), 2)
average_length = round(statistics.mean(lengths), 2)
std_reward = round(statistics.stdev(rewards), 2)
std_length = round(statistics.stdev(lengths), 2)

print(f"Evaluated agent on {agent.env_name} and achieved an average reward of {average_reward} [std: {std_reward}; "
      f"between ({min(rewards)}, {max(rewards)})].\n"
      f"An episode on average took {average_length} steps [std: {std_length}; "
      f"between ({min(lengths)}, {max(lengths)})].\n"
      f"This took me {round(time.time() - start, 2)}s.")
