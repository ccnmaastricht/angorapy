import re

import numpy as np
from matplotlib import pyplot as plt

with open("results/predictability_10000.txt", "r") as f:
    results = [re.match("(.*) from (.*) has an R2 of (.*)", line) for line in f.readlines()]

results = [r.groups() for r in results if r is not None]
result_dict = {}

included_regions = ["M1_activation", "PMC_activation"]
included_targets = ["translation", "translation_to_10", "translation_to_50"]

for r_target, r_activity, r2 in results:
    if r_activity not in result_dict:
        result_dict[r_activity] = {}

    if r_activity not in included_regions:
        continue

    if r_target not in included_targets:
        continue

    result_dict[r_activity][r_target] = float(r2[:-1])

x = np.arange(len(included_targets))
width = 0.35

rects = []
fig, ax = plt.subplots()
ax.bar(x - width / 2,
       [result_dict["M1_activation"][t] for t in included_targets],
       width, label="M1")
ax.bar(x + width / 2,
       [result_dict["PMC_activation"][t] for t in included_targets],
       width, label="PMC")

ax.set_xticks(x, ["next orientation", "orientation in 80ms", "orientation in 400ms"])
ax.legend()
ax.set_ylabel("Predictability (R2)")

ax.set_ylim(0, 1)

# plt.xticks(rotation="vertical")
# plt.gcf().subplots_adjust(bottom=0.3)
plt.show()
