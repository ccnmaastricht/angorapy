import itertools
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

x = np.linspace(0, 1, 100)
for a, b in itertools.product([2, 6, 9], [2, 6, 9]):
    values = stats.beta(a, b).pdf(x)
    plt.plot(x, values, label=f"a={a}; b={b}")

plt.xticks([0, 1])
plt.xlim(0, 1)

fig = plt.gcf()
fig.subplots_adjust(wspace=.5, bottom=0.2)
fig.legend(ncol=3, loc="lower center")
plt.savefig("betas.pdf", format="pdf", bbox_inches='tight')
