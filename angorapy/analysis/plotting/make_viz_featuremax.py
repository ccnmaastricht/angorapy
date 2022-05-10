import os
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.visualization import Visualizer
from utilities.plotting import transparent_cmap

os.chdir("../../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(69420)

# ANALYZERS
analyzer = Visualizer.from_saved_model("storage/pretrained/tanh_c.h5")

# FEATURE MAPS
fids = [1, 2, 3, 4, 5, 6, 7]
fmaxis = analyzer.preferred_stimulus("conv2d_2", feature_ids=fids)

axs: List[Axes]
fig, axs = plt.subplots(1, 7)
fig: Figure
fig.set_size_inches((15, 6))

i = 0
for ax in axs:
    ax.imshow(fmaxis[i], cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    i+=1

plt.savefig("docs/figures/fmaxis.pdf", format="pdf", bbox_inches='tight')
plt.show()
