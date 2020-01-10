#!/usr/bin/env python
"""Load and inspect and agent, scaffold script."""
import os

from agent.ppo import PPOAgent
from utilities.model_management import extract_layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

agent = PPOAgent.from_agent_state(1574852679)

all_weights = agent.policy.get_weights()
print([layer.name for layer in extract_layers(agent.policy)])

print(agent.policy.get_layer("dense").get_weights())
