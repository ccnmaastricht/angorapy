import os

import gym

from agent.ppo import PPOAgent
from analysis.investigation import Investigator
from models import build_rnn_distinct_models, build_ffn_distinct_models, plot_model
from utilities.monitoring import Monitor
import tensorflow as tf

from utilities.util import extract_layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.make("CartPole-v1")
model_builder = build_rnn_distinct_models
agent = PPOAgent(model_builder, env, horizon=1024, workers=8)
monitor = Monitor(agent, env, frequency=1, gif_every=0)

agent.drill(20, 3, 64, monitor=monitor)
agent.save_agent_state()

new_agent = PPOAgent.from_agent_state(agent.agent_id)
investi = Investigator(new_agent.policy)

print(investi.list_layer_names())

plot_model(new_agent.policy, expand_nested=True)

activations_rnn = investi.get_activations_over_episode(investi.list_layer_names()[3], env, True)
print(activations_rnn)

lengths, rewards = agent.evaluate(5, ray_already_initialized=True)
