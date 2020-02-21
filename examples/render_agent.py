#!/usr/bin/env python
"""Example script on loading agent and rendering episodes."""
import os

from agent.ppo import PPOAgent
from analysis.investigation import Investigator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.chdir("../")

AGENT_ID = 1580042580

latest_agent = PPOAgent.from_agent_state(AGENT_ID)
persistent_env = latest_agent.env

# iterate over every save of the agent during training to see evolution of behaviour
for iteration in PPOAgent.get_saved_iterations(AGENT_ID):
    # load agent and wrap an investigator around it
    agent = PPOAgent.from_agent_state(AGENT_ID, from_iteration=iteration)
    inv = Investigator.from_agent(agent)

    # render a randomly initialized episode
    inv.render_episode(persistent_env)

    # just print a line to make output more readable
    print()
