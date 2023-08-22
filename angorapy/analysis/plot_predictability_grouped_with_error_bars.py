import json
import re

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 12}

partial_results = []
for filename in ["./storage/predictability_repeated_august19d.json", "./storage/predictability_repeated_august19e.json"]:
    with open(filename, "r") as f:
        partial_results.append(json.load(f))

results = {}
for key in partial_results[0].keys():
    for target in partial_results[0][key].keys():
        for part in partial_results:
            if key not in part or target not in part[key]:
                continue

            if key not in results:
                results[key] = {}
            if target not in results[key]:
                results[key][target] = []

            results[key][target] = results[key][target] + part[key][target]

included_regions = ["m1_internal", "pmc_recurrent_layer", "goal"]
names = ["M1", "PMC", "Goal"]
included_targets = ["translation", "translation_to_10", "translation_to_50"]
# included_regions = ["M1_activation", "pmc_recurrent_layer"]
# included_targets = ["reward", "goal"]
# included_regions = ["M1_activation", "pmc_recurrent_layer"]
# included_targets = ['rotation_matrix', 'rotation_matrix_last_10']


x = np.arange(len(included_targets))
width = 0.8 / len(included_regions)

rects = []
fig, ax = plt.subplots()

for i, source in enumerate(included_regions, start=-1):
       ax.bar(x + i * width,
              [np.mean(results[source][t]) for t in included_targets],
              width,
              yerr=[np.std(results[source][t]) for t in included_targets],
              label=names[i + 1],
              capsize=3)

ax.set_xticks(x, ["Orientation in 8ms", "Orientation in 80ms", "Orientation in 400ms"])
# ax.set_xticks(x, included_targets)
ax.legend(ncol=3)

ax.set_ylabel("Predictability (R2)")

ax.set_ylim(0, 1)

# plt.show()

# set figure size to 8x2 inches
fig.set_size_inches(8, 2)

# save the figure to a pdf file without whitespace around it
plt.savefig(f"predictability_translation_grouped.pdf", format="pdf", bbox_inches="tight")
