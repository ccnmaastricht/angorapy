#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""
import os
import time

from agent.ppo_agent import PPOAgent
from dexterity.analysis.investigators.base_investigator import Investigator
from common.wrappers import make_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

hrda_id = 1623868261
ra_id = 1623777689
rant_id = 1623748917

id = rant_id

start = time.time()
agent = PPOAgent.from_agent_state(id, "best")
print(f"Agent {id} successfully loaded.")

investigator = Investigator.from_agent(agent)
env = agent.env
env = make_env("NRP" + env.unwrapped.spec.id)

print(f"Evaluating on {env.unwrapped.spec.id} with {env.unwrapped.sim.nsubsteps} substeps.")

for i in range(100):
    investigator.render_episode(env, slow_down=False)
