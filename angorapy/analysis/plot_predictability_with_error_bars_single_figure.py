import json
import re

import numpy as np
from matplotlib import pyplot as plt

with open("results/predictability_repeated_april03.json", "r") as f:
    results = json.load(f)

# remove entries from results that are contain no underscore in their name
results = {k: v for k, v in results.items() if "_" in k}

x = results.keys()

# create x labels by removing 'activation' potentially surrounded by underscores from the x label name
x_labels = [re.sub(r"(_)?activation(_)?", "", s) for s in results.keys()]

# rename x label "pmc_recurrent_layer" to "PMC"
x_labels = [re.sub(r"pmc_recurrent_layer", "PMC", s) for s in x_labels]

targets = ["reward", "goals_achieved_so_far"]

# create a figure with a single plot
fig, axs = plt.subplots()

# plot barplots for all targets, with bars placed next to each other not on top of each other
for i, target in enumerate(targets):
    # make bar width adaptive to number of targets
    bar_width = 1 / (len(targets)) - 0.1

    # plot barplot for target
    axs.bar(np.arange(len(x)) + i * bar_width - bar_width,
            [np.mean(results[s][target]) for s in results.keys()],
            width=bar_width,
            yerr=[np.std(results[s][target]) for s in results.keys()],
            capsize=2,
            label=target)

    # set labels of the x axis to keys in results
    axs.set_xticks(np.arange(len(x)))
    axs.set_xticklabels(x_labels)

    axs.set_ylabel("Predictability (R2)")
    axs.set_ylim(0, 1)

    # add legend above figure based on targets
    axs.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

plt.show()

# for target in possible_targets:
#     fig, ax = plt.subplots()
#     ax.bar(x,
#        [np.mean(results[s][target]) for s in results.keys()],
#        yerr=[np.std(results[s][target]) for s in results.keys()],
#        capsize=3)
#
#     ax.set_title(f"Predicting {target}")
#
#
#
#     plt.xticks(rotation="vertical")
#     plt.gcf().subplots_adjust(bottom=0.3)
#     plt.show()
