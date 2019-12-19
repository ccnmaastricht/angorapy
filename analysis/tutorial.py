import os

import gym
# import tensorflow as tf
from agent.ppo import PPOAgent
# from analysis.investigation import Investigator
from models import build_rnn_distinct_models, build_ffn_distinct_models
from utilities.monitoring import Monitor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.make("LunarLanderContinuous-v2")
#model_builder = build_rnn_distinct_models
model_builder = build_rnn_distinct_models
agent = PPOAgent(model_builder, env, horizon=1024, workers=8)
monitor = Monitor(agent, env, 25, 25)

agent.drill(200, 10, 64, monitor=monitor)
agent.save_agent_state()

new_agent = PPOAgent.from_agent_state(agent.agent_id)
#investi = Investigator(new_agent.policy)
print(agent.agent_id)
#print(investi.list_layer_names())
#print(investi.list_layer_names()[3])

# activations_lstm = investi.get_activations_over_episode(investi.list_layer_names()[2], env, True)
# print(activations_lstm)

# lengths, rewards = new_agent.evaluate(5, ray_already_initialized=True)