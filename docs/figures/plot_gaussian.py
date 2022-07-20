import itertools
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

x = np.linspace(0, 1, 100)
for a, b in itertools.product([0.3, 0.5, 0.7], [0.05, 0.1, 0.3]):
    values = stats.norm(a, b).pdf(x)
    plt.plot(x, values, label=f"a={a}; b={b}")

plt.xticks([0, 1])
plt.xlim(0, 1)

plt.xlabel("Action")
plt.ylabel("PDF")

fig = plt.gcf()
fig.subplots_adjust(wspace=.5, bottom=0.25)
fig.legend(ncol=3, loc="lower center")
plt.savefig("gaussians.pdf", format="pdf", bbox_inches='tight')
plt.show()
