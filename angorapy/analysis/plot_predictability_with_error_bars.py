import json
import re

import numpy as np
from matplotlib import pyplot as plt

with open("results/predictability_repeated.json", "r") as f:
    results = json.load(f)

target = "fingertip_positions"
print(f"Possible targets: {list(list(results.items())[0][1].keys())}")

x = results.keys()

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
