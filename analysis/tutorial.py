import os

import gym
import tensorflow as tf


from agent.ppo import PPOAgent
from analysis.investigation import Investigator
from models import build_rnn_distinct_models
from utilities.monitoring import Monitor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.make("CartPole-v1")
model_builder = build_rnn_distinct_models
agent = PPOAgent(model_builder, env, horizon=1024, workers=2)
monitor = Monitor(agent, env, 1)

agent.drill(2, 3, 2, monitor=monitor)
agent.save_agent_state()

new_agent = PPOAgent.from_agent_state(agent.agent_id)
investi = Investigator(new_agent.policy)

print(investi.list_layer_names())
print(investi.list_layer_names()[2])

activations_lstm = investi.get_activations_over_episode(investi.list_layer_names()[2], env, True)
print(activations_lstm)

lengths, rewards = new_agent.evaluate(5, ray_already_initialized=True)