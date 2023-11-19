import json
import re

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from angorapy.common.const import QUALITATIVE_COLOR_PALETTE as COLORS

font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


with open("storage/predictability_repeated_august19d.json", "r") as f:
    results = json.load(f)



included_regions = ["m1_internal", "pmc_recurrent_layer", "goal"]
included_targets = ["translation", "translation_to_10", "translation_to_20", "translation_to_30", "translation_to_40", "translation_to_50"]
# included_regions = ["M1_activation", "pmc_recurrent_layer"]
# included_targets = ["reward", "goal"]
# included_regions = ["M1_activation", "pmc_recurrent_layer"]
# included_targets = ['rotation_matrix', 'rotation_matrix_last_10']


x = [8 * x for x in [1, 10, 20, 30, 40, 50]]

m1_means = [np.mean(results["m1_internal"][t]) for t in included_targets]
pmc_means = [np.mean(results["pmc_recurrent_layer"][t]) for t in included_targets]
goal_means = [np.mean(results["goal"][t]) for t in included_targets]

m1_stds = [np.std(results["m1_internal"][t]) for t in included_targets]
pmc_stds = [np.std(results["pmc_recurrent_layer"][t]) for t in included_targets]
goal_stds = [np.std(results["goal"][t]) for t in included_targets]

rects = []
fig, ax = plt.subplots()

ax.plot(x, m1_means, label="M1", marker=None, color=COLORS[0])
ax.plot(x, pmc_means, label="PMC", marker=None, color=COLORS[1])
ax.plot(x, goal_means, label="Goal", marker=None, linestyle="--", alpha=0.5, color=COLORS[2])

# add y error bars in same colors as mean bars
ax.errorbar(x, m1_means, yerr=m1_stds, fmt='none', ecolor=COLORS[0], capsize=2)
ax.errorbar(x, pmc_means, yerr=pmc_stds, fmt='none', ecolor=COLORS[1], capsize=2)

ax.set_xlabel("Orientation in X ms")
ax.legend(ncol=3)

ax.set_ylabel("Predictability (R2)")

ax.set_ylim(0, 1)

# plt.show()

# set figure size to 8x2 inches
fig.set_size_inches(8, 3)
plt.tight_layout()

# save the figure to a pdf file without whitespace around it
plt.savefig(f"predictability_future_translation.pdf", format="pdf", bbox_inches="tight")
