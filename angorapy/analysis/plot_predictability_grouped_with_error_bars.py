import json
import re

import numpy as np
from matplotlib import pyplot as plt

with open("results/predictability_repeated_mar20.json", "r") as f:
    results = json.load(f)

included_regions = ["M1_activation", "pmc_recurrent_layer", "goal"]
included_targets = ["translation", "translation_to_10", "translation_to_50"]
# included_regions = ["M1_activation", "pmc_recurrent_layer"]
# included_targets = ["reward", "goal"]
# included_regions = ["M1_activation", "pmc_recurrent_layer"]
# included_targets = ['rotation_matrix', 'rotation_matrix_last_10']


x = np.arange(len(included_targets))
width = 0.7 / len(included_regions)

rects = []
fig, ax = plt.subplots()

for i, source in enumerate(included_regions):
       ax.bar(x - (width / (len(included_regions)) * i),
              [np.mean(results[source][t]) for t in included_targets],
              width,
              yerr=[np.std(results[source][t]) for t in included_targets],
              label=source.split("_")[0],
              capsize=3)

# ax.set_xticks(x, ["orientation in 8ms", "orientation in 80ms", "orientation in 400ms"])
ax.set_xticks(x, included_targets)
ax.legend()
ax.set_ylabel("Predictability (R2)")

ax.set_ylim(0, 1)

plt.show()

# plt.xticks(rotation="vertical")
# plt.gcf().subplots_adjust(bottom=0.3)
# plt.savefig("M1vsPMC.svg", format="svg", bbox_inches="tight")
