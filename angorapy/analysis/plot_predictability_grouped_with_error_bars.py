import json
import re

import numpy as np
from matplotlib import pyplot as plt

with open("results/predictability_repeated.json", "r") as f:
    results = json.load(f)

included_regions = ["M1_activation", "PMC_activation"]
included_targets = ["translation", "translation_to_10", "translation_to_50"]

x = np.arange(len(included_targets))
width = 0.35

rects = []
fig, ax = plt.subplots()
ax.bar(x - width / 2,
       [np.mean(results["M1_activation"][t]) for t in included_targets],
       width,
       yerr=[np.std(results["M1_activation"][t]) for t in included_targets],
       label="M1",
       capsize=3)
ax.bar(x + width / 2,
       [np.mean(results["PMC_activation"][t]) for t in included_targets],
       width,
       yerr=[np.std(results["PMC_activation"][t]) for t in included_targets],
       label="PMC",
       capsize=3)

ax.set_xticks(x, ["orientation in 8ms", "orientation in 80ms", "orientation in 400ms"])
ax.legend()
ax.set_ylabel("Predictability (R2)")

ax.set_ylim(0, 1)

# plt.xticks(rotation="vertical")
# plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig("M1vsPMC.svg", format="svg", bbox_inches="tight")
