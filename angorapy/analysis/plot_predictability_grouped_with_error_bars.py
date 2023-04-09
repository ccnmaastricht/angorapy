import json
import re

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
with open("../../storage/predictability_repeated_april08.json", "r") as f:
    results = json.load(f)

included_regions = ["M1_activation", "pmc_recurrent_layer", "goal"]
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
              label=source.split("_")[0].upper() if "_" in source else source.capitalize(),
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
