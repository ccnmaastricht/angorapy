import os
import random
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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.random.set_seed(69420)

# load the input image
reference = mpimg.imread("analysis/img/hand.png")

# ANALYZERS
analyzer_c = Visualizer.from_saved_model("storage/pretrained/tanh_c.h5", mode="return")
analyzer_r = Visualizer.from_saved_model("storage/pretrained/tanh_r.h5", mode="return")
analyzer_p = Visualizer.from_saved_model("storage/pretrained/tanh_h.h5", mode="return")

axs: List[Axes]
fig: Figure = plt.figure(figsize=(9, 3))
grid = plt.GridSpec(2, 6)

ax: Axes
for row in [0, 1]:
    for col in [0, 1, 2, 3, 4, 5]:
        analyzer = analyzer_c
        if col > 1:
            analyzer = analyzer_r
        if col > 3:
            analyzer = analyzer_p

        salience = analyzer.saliency_map(reference, layer_name="dense_1", neuron=random.randint(0, 512))

        print([row, col])
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(salience, cmap="jet")
        ax.set_xticks([])
        ax.set_yticks([])

plt.savefig("docs/figures/pretraining_methods_saliency.pdf", format="pdf", bbox_inches='tight')
plt.show()
