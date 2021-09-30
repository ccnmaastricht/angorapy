from typing import List, Dict

from bokeh import embed
from bokeh.plotting import figure

from utilities.monitor.plotting_base import palette, plot_styling, style_plot
import numpy as np
import pandas as pd


def plot_episode_box_plots(rewards: List[float], lengths: List[float]):
    """Boxplot of the rewards and lengths given."""
    p = figure(title="Preprocessor Running Mean",
               x_axis_label='Cycle',
               y_axis_label='Running Mean',
               **plot_styling)

    cats = ["reward"]
    data = np.stack([rewards])

    # find the quartiles and IQR for each category
    q1 = np.quantile(data, q=0.25, axis=-1)
    q2 = np.quantile(data, q=0.5, axis=-1)
    q3 = np.quantile(data, q=0.75, axis=-1)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # find the outliers for each category
    # def outliers(group):
    #     cat = group.name
    #     return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    #
    # out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    # if not out.empty:
    #     outx = list(out.index.get_level_values(0))
    #     outy = list(out.values)

    p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)

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
    p.circle(cats * subsamplesize, np.random.choice(data[0], subsamplesize), size=6, color="#F38630", fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size = "16px"

    p.legend.location = "bottom_right"
    style_plot(p)

    return embed.components(p)