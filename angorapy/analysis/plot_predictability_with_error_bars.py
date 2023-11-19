import json
import re

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
with open("./storage/predictability_repeated_august19a.json", "r") as f:
    results = json.load(f)

# remove entries from results that are contain no underscore in their name
results = {k: v for k, v in results.items() if "_" in k}

# create x labels by removing 'activation' potentially surrounded by underscores from the x label name
x_labels = [re.sub(r"(_)?activation(_)?", "", s) for s in results.keys()]
x_labels = [re.sub(r"(_)?internal(_)?", "", s) for s in x_labels]

# rename x label "pmc_recurrent_layer" to "PMC"
x_labels = [re.sub(r"pmc_recurrent_layer", "PMC", s) for s in x_labels]

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
    ax.set_title(f"{target}")

    ax.set_ylim(0, 1)

    # set labels of the x axis to keys in results
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x_labels)

    # show x labels vertically
    plt.xticks(rotation="vertical")

    # set figure size to 8x2 inches
    fig.set_size_inches(12, 2)

    # plt.show()

    # save the figure to a pdf file without whitespace around it
    plt.savefig(f"predictability_{target}.pdf", format="pdf", bbox_inches="tight")