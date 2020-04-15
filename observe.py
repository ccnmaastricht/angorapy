#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""
import argparse
import os
import time

from agent.ppo import PPOAgent
from analysis.investigation import Investigator
from utilities.const import BASE_SAVE_PATH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=None)
args = parser.parse_args()

if args.id is None:
    ids = map(int, os.listdir(BASE_SAVE_PATH))
    args.id = max(ids)

start = time.time()
agent = PPOAgent.from_agent_state(args.id, "b")
print(f"Agent {args.id} successfully loaded.")

investigator = Investigator.from_agent(agent)

for i in range(100):
    investigator.render_episode(agent.env, slow_down=True)
