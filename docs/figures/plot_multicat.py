import itertools
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

# setup the figure and axes
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')

# fake data
_x = np.arange(20)
_y = np.arange(11)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = np.random.random(x.shape)
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_xlabel("Action")
ax1.set_ylabel("Bin")
ax1.set_zlabel("Probability")

plt.savefig("multicat.pdf", format="pdf", bbox_inches='tight')
plt.show()
