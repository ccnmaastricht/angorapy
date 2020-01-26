#!/usr/bin/env python
"""Example script on loading and inspecting an agent."""
import os

from agent.ppo import PPOAgent
from analysis.investigation import Investigator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.chdir("../")

agent = PPOAgent.from_agent_state(1580042580)
inv = Investigator.from_agent(agent)

# render agent at different steps
inv.render_episode(agent.env)
