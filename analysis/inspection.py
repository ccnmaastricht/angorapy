#!/usr/bin/env python
"""Load and inspect and agent, scaffold script."""
import os

from agent.ppo import PPOAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

agent = PPOAgent.from_agent_state(1574781207)

all_weights = agent.joint.get_weights()

print(all_weights)
