#!/usr/bin/env python
import argparse
import json

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.analysis import investigators

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1679142835973298)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")
parser.add_argument("--repeats", type=str, help="how many models per setting", default=10)

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../")
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

targets = ["proprioception", "vision", "touch", "reward", "fingertip_positions", "goal", "translation",
           "translation_to_10", "translation_to_50", "object_orientation", "rotation_matrix",
           "rotation_matrix_last_10", "current_rotational_axis"]
results = {
    "noise": {},
    "proprioception": {},
    "vision": {},
    "touch": {},
    "goal": {},
    "SSC_activation_1": {},
    "SSC_activation_2": {},
    "LPFC_activation": {},
    "MCC_activation": {},
    "IPL_activation": {},
    "SPL_activation": {},
    "IPS_activation": {},
    "M1_activation": {},
    "lstm_cell_1": {},
    "pmc_recurrent_layer": {},
}

for information in targets:
    for source in results.keys():
        results[source][information] = []

for i in range(args.repeats):
    print(f"Collecting for repeat {i + 1}/{args.repeats}")
    investigator.prepare(env,
                         layers=[
                             "SSC_activation_1",
                             "SSC_activation_2",
                             "LPFC_activation",
                             "MCC_activation",
                             "IPL_activation",
                             "SPL_activation",
                             "IPS_activation",
                             "pmc_recurrent_layer",
                             "lstm_cell_1",
                             "M1_activation",
                         ],
                         n_states=20000)

    for information in targets:
        print(f"Predicting {information} from activity.\n-----------------------")

        for source in results.keys():
            results[source][information].append(investigator.fit(source, information))

        print("\n\n")

with open("analysis/results/predictability_repeated_mar21.json", "w") as f:
    json.dump(results, f)
