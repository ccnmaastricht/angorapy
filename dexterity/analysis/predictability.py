#!/usr/bin/env python
import argparse

import tensorflow as tf
import tqdm

from dexterity.agent.ppo_agent import PPOAgent
from dexterity.analysis.investigation import Investigator
from dexterity.utilities.hooks import register_hook
from dexterity.utilities.model_utils import is_recurrent_model
from dexterity.utilities.util import flatten

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1639582562)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../../")
investigator = Investigator.from_agent(agent)
env = agent.env

is_recurrent = is_recurrent_model(investigator.network)
investigator.network.reset_states()

done, step = False, 0
state = env.reset()
state_collection = []
activation_collection = []
n_states = 10
for i in tqdm.tqdm(range(n_states)):
    state_collection.append(state)

    prepared_state = state.with_leading_dims(time=is_recurrent).dict()
    probabilities = flatten(investigator.network(prepared_state, training=False))
    action, _ = investigator.distribution.act(*probabilities)

    observation, reward, done, info = env.step(action)

    if done:
        state = env.reset()
    else:
        state = observation

prepared_state = state.with_leading_dims(time=is_recurrent).dict()

activations = investigator.get_layer_activations(
    ['SomatosensoryCortex', 'PrefrontalCortex', 'PosteriorParietalCortex', 'MotorCortex'],
    prepared_state
)
print(activations.keys())

activations_dataset = tf.data.Dataset.from_tensor_slices()

# train predictor on information extraction
predictor = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(100),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(100),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
predictor.compile(optimizer)

# predictor.fit(
#
# )
