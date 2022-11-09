import json
import statistics

import numpy as np
from matplotlib import pyplot as plt

from angorapy.common.const import PATH_TO_EXPERIMENTS

ids = ['1667916362968118', '1667916362368835', '1667916362351224', '1667916362238643', '1667830229410560',
       '1667829693813697', '1667829690921903', '1667829690849747', '1667829690803225', '1667829689265280']

n_workers = []
n_optimizers = []
avg_opt = []
avg_gat = []


for id in ids:
    with open(f"../../../{PATH_TO_EXPERIMENTS}/{id}/meta.json", "r") as f:
        meta = json.load(f)
        n_workers.append(meta["n_cpus"])
        n_optimizers.append(meta["n_gpus"])

    with open(f"../../../{PATH_TO_EXPERIMENTS}/{id}/statistics.json", "r") as f:
        stats = json.load(f)
        avg_opt.append(statistics.mean(stats["optimization_timings"]))
        avg_gat.append(statistics.mean(stats["gathering_timings"]))

zippedi = list(zip(n_workers, n_optimizers, avg_opt, avg_gat))
zippedi.sort()
n_workers, n_optimizers, avg_opt, avg_gat = zip(*zippedi)

fig, ax = plt.subplots()
ax_opt = ax.twiny()

avg_gat = (np.array(avg_gat) - np.min(avg_gat)) / (np.max(avg_gat) - np.min(avg_gat))
avg_opt = (np.array(avg_opt) - np.min(avg_opt)) / (np.max(avg_opt) - np.min(avg_opt))

ax.plot(n_workers, avg_gat)
ax_opt.plot(n_optimizers, avg_opt, color="red")

plt.show()