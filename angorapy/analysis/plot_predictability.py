import re

from matplotlib import pyplot as plt

with open("results/predictability_.txt", "r") as f:
    results = [re.match("(.*) from (.*) has an R2 of (.*)", line) for line in f.readlines()]

results = [r.groups() for r in results if r is not None]
result_dict = {}

for r_target, r_activity, r2 in results:
    if r_target not in result_dict:
        result_dict[r_target] = {}

    result_dict[r_target][r_activity] = float(r2[:-1])


for target, activity_r2s in result_dict.items():
    source_names = activity_r2s.keys()
    source_r2s = [activity_r2s[n] for n in source_names]
    plt.bar(source_names, source_r2s)
    plt.title(target)
    plt.xticks(rotation="vertical")
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.show()
