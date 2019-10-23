#!/usr/bin/env python
"""Example script for evaluating a loaded agent on a task. """
import os
import statistics

from agent.ppo import PPOAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


AGENT_ID = 1571837490

agent = PPOAgent.from_agent_state(AGENT_ID)
print(f"Agent {AGENT_ID} successfully loaded.")

rewards = agent.evaluate(10)

average_reward = round(statistics.mean(rewards), 2)
std_reward = round(statistics.stdev(rewards), 2)

print(f"Evaluated agent on {agent.env_name} and achieved an average reward of {average_reward} [std: {std_reward}].")
