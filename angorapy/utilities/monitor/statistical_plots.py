from typing import List, Dict

import bokeh.layouts
from bokeh import embed
from bokeh.plotting import figure

from angorapy.utilities.monitor.plotting_base import palette, plot_styling, style_plot
import numpy as np
import pandas as pd


def plot_episode_box_plots(rewards: List[float], lengths: List[float]):
    """Boxplot of the rewards and lengths given."""
    if len(rewards) < 1:
        return "", ""

    cats = ["reward"]
    rewards = np.array(rewards, dtype=np.float)

    # find the quartiles and IQR for each category
    q1 = np.quantile(rewards, q=0.25, axis=-1)
    q2 = np.quantile(rewards, q=0.5, axis=-1)
    q3 = np.quantile(rewards, q=0.75, axis=-1)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # find the outliers for each category
    # def outliers(group):
    #     cat = group.name
    #     return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    #
    # out = groups.apply(outliers).dropna()

    # prepare outlier rewards for plotting, we need coordinates for every outlier.
    # if not out.empty:
    #     outx = list(out.index.get_level_values(0))
    #     outy = list(out.values)

    p = figure(title="Preprocessor Running Mean",
               x_axis_label='Cycle',
               y_axis_label='Running Mean',
               **plot_styling,
               x_range=cats
    )

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    # qmin = groups.quantile(q=0.00)
    # qmax = groups.quantile(q=1.00)
    # upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
    # lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]

    # stems
    p.segment(cats, upper, cats, q3, line_color="black")
    p.segment(cats, lower, cats, q1, line_color="black")

    # boxes
    p.vbar(cats, 0.7, q2, q3, fill_color="#E08E79", line_color="black")
    p.vbar(cats, 0.7, q1, q2, fill_color="#3B8686", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower, 0.2, 0.01, line_color="black")
    p.rect(cats, upper, 0.2, 0.01, line_color="black")

    # outliers
    # if not out.empty:
    #     p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    subsamplesize = 100
    if len(rewards) > subsamplesize:
        rewards = np.random.choice(rewards, subsamplesize)

    p.circle(cats * len(rewards), rewards, size=6, color="#F38630", fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size = "16px"

    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def plot_per_receptor_mean(per_receptor_mean: Dict[str, List]):
    plots = []
    for i, (sense, means) in enumerate(per_receptor_mean.items()):
        p = figure(title=f"{sense.capitalize()} Distribution",
                   x_axis_label='Receptor',
                   y_axis_label='Mean',
                   **plot_styling,
                   )

        p.vbar(range(len(means)), width=1, top=means, legend_label=sense, color=palette[i])
        p.legend.location = "bottom_right"
        style_plot(p)

        plots.append(p)

    p = bokeh.layouts.column(plots)
    p.sizing_mode = "stretch_width"
    return p
