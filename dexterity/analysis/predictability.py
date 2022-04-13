#!/usr/bin/env python
import argparse

from dexterity.agent.ppo_agent import PPOAgent
from dexterity.analysis.investigation import Investigator
from dexterity.utilities.hooks import register_hook
from dexterity.utilities.model_utils import is_recurrent_model

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
prepared_state = state.with_leading_dims(time=is_recurrent).dict()
investigator.network(prepared_state)

activations = investigator.get_layer_activations(["SomatosensoryCortex"], prepared_state)

print(activations.keys())