#!/usr/bin/env python
import argparse

from keras.utils import plot_model, model_to_dot

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.analysis import investigators
import networkx as nx


parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1653053413)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../../")
investigator = investigators.Predictability.from_agent(agent)
env = agent.env

# nx_model = nx.drawing.nx_pydot.from_pydot(model_to_dot(investigator.network, expand_nested=True, subgraph=True))
#
# model_to_dot(investigator.network, expand_nested=True).write_png("dot_sankey.png")

# nodes_to_delete = []
# for node, data in zip(nx_model.nodes(), list(nx_model.nodes.values())):
#     if data.get("label") is None or "mask" in data["label"]:
#         nodes_to_delete.append(node)

# for node_name in nodes_to_delete:
#     nx_model.add_edge(*nx_model.predecessors(node_name), *nx_model.successors(node_name))
#     nx_model.remove_node(node_name)

# make output
# nx.drawing.nx_pydot.to_pydot(nx_model).write_png("sankey.png")

investigator.prepare(env,
                     layers=[
                         "SSC_activation_1",
                         "SSC_activation_2",
                         "LPFC_activation",
                         "MCC_activation",
                         "IPL_activation",
                         "SPL_activation",
                         "IPS_activation",
                         "PMC_activation",
                         "M1_activation",
                     ],
                     n_states=2000)


for information in ["proprioception", "vision", "somatosensation", "reward", "fingertip_positions", "goal", "translation",
                    "translation_to_10", "translation_to_50", "object_orientation", "translation_matrix"]:
    print(f"Predicting {information} from activity.\n-----------------------")

    investigator.measure("noise", information)
    investigator.measure("SSC_activation_1", information)
    investigator.measure("SSC_activation_2", information)
    investigator.measure("LPFC_activation", information)
    investigator.measure("MCC_activation", information)
    investigator.measure("IPL_activation", information)
    investigator.measure("SPL_activation", information)
    investigator.measure("IPS_activation", information)
    investigator.measure("PMC_activation", information)
    investigator.measure("M1_activation", information)
    investigator.measure("proprioception", information)
    investigator.measure("vision", information)
    investigator.measure("somatosensation", information)

    print("\n\n")
