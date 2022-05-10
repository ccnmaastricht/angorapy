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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.random.set_seed(69420)

# load the input image
reference = mpimg.imread("img/hand.png")
reference_gray = tf.squeeze(tf.image.rgb_to_grayscale(reference)).numpy()
hm_cmap = transparent_cmap(plt.cm.get_cmap("hot"))

# ANALYZERS
analyzer_c = Visualizer.from_saved_model("storage/pretrained/caltech_c.h5", mode="return")
analyzer_r = Visualizer.from_saved_model("storage/pretrained/visual_r.h5", mode="return")
analyzer_p = Visualizer.from_saved_model("storage/pretrained/hands_h.h5", mode="return")

# FEATURE MAPS
fm_heat_c = analyzer_c.feature_map("conv2d_3", reference, mode="heat")
fm_heat_r = analyzer_r.feature_map("conv2d_3", reference, mode="heat")
fm_heat_p = analyzer_p.feature_map("conv2d_3", reference, mode="heat")

plt.close()

axs: List[Axes]
fig: Figure = plt.figure(figsize=(12, 3))
grid = plt.GridSpec(2, 8)

reference_ax = fig.add_subplot(grid[:, :2])
reference_ax.imshow(reference)
reference_ax.set_xticks([]), reference_ax.set_yticks([])

filter_ids = [[10, 14], [17, 27]]

ax: Axes
for row in [0, 1]:
    row_filter_ids = filter_ids[row]
    for col in [2, 3, 4, 5, 6, 7]:
        fi = row_filter_ids[col % 2]
        fm = fm_heat_c
        t_color = "red"
        if col > 3:
            fm = fm_heat_r
            t_color = "green"
        if col > 5:
            fm = fm_heat_p
            t_color = "blue"

        ax = fig.add_subplot(grid[row, col])
        ax.set_facecolor(t_color)
        ax.imshow(reference_gray, cmap="gray")
        ax.contourf(fm[:, :, fi], cmap=hm_cmap)
        ax.set_xticks([])
        ax.set_yticks([])

plt.savefig("docs/figures/pretraining_methods_fms.pdf", format="pdf", bbox_inches='tight')
plt.show()
