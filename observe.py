#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""
import argparse
import os
import time

import gym

from agent.ppo import PPOAgent
from analysis.investigation import Investigator
from utilities.const import BASE_SAVE_PATH
from common.wrappers import make_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=None)
parser.add_argument("--env", type=str, nargs="?", help="force testing environment", default="")
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="b")
parser.add_argument("--force-case-circulation", action="store_true", help="circle through goal definitions")
parser.add_argument("--rcon", type=str, help="reward configuration", default=None)

args = parser.parse_args()

if args.state not in ["b", "best"]:
    args.state = int(args.state)

if args.id is None:
    ids = map(int, [d for d in os.listdir(BASE_SAVE_PATH) if os.path.isdir(os.path.join(BASE_SAVE_PATH, d))])
    args.id = max(ids)

start = time.time()
agent = PPOAgent.from_agent_state(args.id, args.state, force_env_name=None if not args.env else args.env)
print(f"Agent {args.id} successfully loaded.")

investigator = Investigator.from_agent(agent)
env = agent.env
if args.env != "":
    env = make_env(args.env, args.rcon)

print(f"Evaluating on {env.unwrapped.spec.id}")

if not args.force_case_circulation or (env.unwrapped.spec.id != "FreeReachAbsolute-v0"):
    for i in range(100):
        investigator.render_episode(env, slow_down=False)
else:
    env = gym.make("FreeReachFFAbsolute-v0")
    for i in range(100):
        env.forced_finger = i % 4
        env.env.forced_finger = i % 4
        env.unwrapped.forced_finger = i % 4
        investigator.render_episode(env, slow_down=False)