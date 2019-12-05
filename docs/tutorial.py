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
model_builder = build_ffn_distinct_models
agent = PPOAgent(model_builder, env, horizon=1024, workers=8)
monitor = Monitor(agent, env, 5)

# agent.drill(2, 3, 64, monitor=monitor)
agent.save_agent_state()

# agent = PPOAgent.from_agent_state(agent.agent_id)
investi = Investigator(agent.policy)

print(list(map(lambda x: x.name, agent.policy.layers)))
print(investi.list_layer_names())
# print(investi.list_layer_names()[3])

plot_model(agent.policy, expand_nested=True)
for layer in extract_layers(agent.policy):
    try:
        intermediate_model = tf.keras.Model(inputs=[agent.policy.input], outputs=[agent.policy.get_layer("policy_encoder").get_layer("dense").output])
    except ValueError as err:
        print(layer.name + " did not work")
        raise err
    else:
        print(layer.name + " worked")
# activations_lstm = investi.get_activations_over_episode(investi.list_layer_names()[3], env, True)
# print(activations_lstm)

# lengths, rewards = agent.evaluate(5, ray_already_initialized=True)
