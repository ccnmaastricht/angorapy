import json
import re

import numpy as np
from matplotlib import pyplot as plt

with open("results/predictability_repeated_mar20.json", "r") as f:
    results = json.load(f)

x = results.keys()
possible_targets = list(list(results.items())[0][1].keys())
print(f"Possible targets: {possible_targets}")

for target in possible_targets:
    fig, ax = plt.subplots()
    ax.bar(x,
       [np.mean(results[s][target]) for s in results.keys()],
       yerr=[np.std(results[s][target]) for s in results.keys()],
       capsize=3)

    ax.set_ylabel("Predictability (R2)")
    ax.set_title(f"Predicting {target}")

    ax.set_ylim(0, 1)

    plt.xticks(rotation="vertical")
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.show()
