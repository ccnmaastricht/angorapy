import json

import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("exp", nargs="?", type=str, default="cheetah", help="name of the experiment")

args = parser.parse_args()

with open(f"../../docs/benchmarks/{args.exp}.json") as f:
    results = json.load(f)

x = list(range(1, len(results[list(results.keys())[0]][0]) + 1))

for name in results:
    sub_exp = results[name]

    plt.plot(x, sub_exp[0], 'k-')
    plt.fill_between(x, np.array(sub_exp[0]) - np.array(sub_exp[1]),
                     np.array(sub_exp[0]) + np.array(sub_exp[1]),
                     label=f"{name}")

    plt.legend()
    plt.title(f"Benchmarking {args.exp}")
    plt.savefig(f"../../docs/figures/benchmarking_{args.exp}.pdf", format="pdf")
