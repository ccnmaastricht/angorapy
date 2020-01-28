import json

import matplotlib.pyplot as plt
import argparse
import numpy as np

from utilities.plotting import plot_with_confidence

colors = ["red", "green", "blue", "orange"]

parser = argparse.ArgumentParser()
parser.add_argument("exp", nargs="*", type=str, default="LunarLanderContinuous_continuous_continuous_beta", help="name of the experiment")

args = parser.parse_args()

results = {}
for exp in args.exp:
    with open(f"docs/benchmarks/{exp}.json") as f:
        results.update(json.load(f))

x = list(range(1, len(results[list(results.keys())[0]][0]) + 1))

for i, name in enumerate(results):
    sub_exp = results[name]

    plot_with_confidence(x=sub_exp[0],
                         lb=np.array(sub_exp[0]) - np.array(sub_exp[1]),
                         ub=np.array(sub_exp[0]) + np.array(sub_exp[1]),
                         label=name,
                         col=colors[i])

plt.legend()
plt.title(f"Benchmarking {args.exp}")
plt.savefig(f"docs/benchmarks/benchmarking_{args.exp}.pdf", format="pdf")
plt.show()
