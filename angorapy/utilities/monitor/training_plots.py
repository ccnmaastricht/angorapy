from typing import List, Dict, Tuple

import bokeh
import numpy
import numpy as np
import pandas as pd
import scipy
from bokeh import embed, layouts
from bokeh.io import export_svg
from bokeh.models import Span, Range1d, LinearAxis, ColumnDataSource, HoverTool
from bokeh.plotting import figure
from scipy import stats

from angorapy.utilities.monitor.plotting_base import palette, plot_styling, style_plot
from angorapy.utilities.statistics import mean_fill_nones


def plot_memory_usage(memory_trace: List):
    """Plot the development of memory used by the training."""
    x = list(range(len(memory_trace)))
    y = memory_trace

    p = figure(title="Memory Usage",
               x_axis_label='Cycle',
               y_axis_label='Memory in GB',
               y_range=(min(memory_trace), max(memory_trace)),
               **plot_styling)

    p.line(x, y, legend_label="RAM", line_width=2, color=palette[0])

    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def plot_execution_times(cycle_timings, optimization_timings=None, gathering_timings=None):
    """Plot the execution times of a full cycle and optionally bot sub phases."""
    x = list(range(len(cycle_timings)))

    all_times = (cycle_timings
                 + (optimization_timings if optimization_timings is not None else [])
                 + (gathering_timings if gathering_timings is not None else []))

    if len(all_times) < 1:
        return None, None

    p = figure(title="Execution Times",
               x_axis_label='Cycle',
               y_axis_label='Seconds',
               y_range=(0, max(all_times)),
               **plot_styling)

    p.line(x, cycle_timings, legend_label="Full Cycle", line_width=2, color=palette[0])
    if optimization_timings is not None:
        p.line(x, optimization_timings, legend_label="Optimization Phase", line_width=2, color=palette[1])
    if gathering_timings is not None:
        p.line(x, gathering_timings, legend_label="Gathering Phase", line_width=2, color=palette[2])

    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def plot_reward_progress(rewards: Dict[str, list], cycles_loaded, reward_threshold=None):
    """Plot the execution times of a full cycle and optionally bot sub phases."""
    means, stds = rewards["mean"], rewards["stdev"]
    stds = np.array(stds)
    stds[stds == None] = 0
    stds = stds * 0.2
    means = np.array(means)
    means = mean_fill_nones(means)

    x = list(range(len(means)))
    df = pd.DataFrame(data=dict(x=x, y=means, lower=np.subtract(means, stds), upper=np.add(means, stds)))
    value_range = max(df["upper"]) - min(df["lower"])

    tooltip = HoverTool(
        tooltips=[("Cycle", "@x"),
                  ("Reward", "@y")],
        mode="vline"
    )

    range_max = max(df["upper"]) + value_range * 0.2
    if reward_threshold is not None:
        range_max = max(range_max, reward_threshold * 1.1)
    p = figure(title="Average Rewards per Cycle",
               x_axis_label='Cycle',
               y_axis_label='Total Episode Return',
               y_range=(min(df["lower"]), range_max),
               x_range=(0, max(x)),
               **plot_styling)

    p.add_tools(tooltip)
    p.add_tools(bokeh.models.BoxZoomTool())
    p.add_tools(bokeh.models.ResetTool())
    p.add_tools(bokeh.models.SaveTool())

    # ERROR BAND
    error_band = bokeh.models.Band(
        base="x", lower="lower", upper="upper",
        source=ColumnDataSource(df.reset_index()),
        fill_color=palette[0],
        fill_alpha=0.2,
        line_color=palette[0],
        line_alpha=0.4,
    )
    p.add_layout(error_band)
    p.renderers.extend([error_band])

    # REWARD LINE
    p.line(x, means, legend_label="Reward", line_width=2, color=palette[0])

    # REWARD THRESHOLD
    if reward_threshold is not None:
        p.line(x, [reward_threshold for _ in x], line_color="green", line_width=2, line_alpha=0.7, line_dash="dashed",
               legend_label="Solution Threshold")

    # MAX VALUE MARKING
    x_max = np.argmax(means)
    y_max = np.max(means)
    p.add_layout(bokeh.models.Arrow(end=bokeh.models.NormalHead(size=10,
                                                                line_color="darkred",
                                                                line_width=2,
                                                                fill_color="red"),
                                    line_color="darkred",
                                    line_width=2,
                                    x_start=x_max, y_start=y_max + value_range * 0.07,
                                    x_end=x_max, y_end=y_max))
    p.add_layout(bokeh.models.Label(x=x_max, y=y_max + value_range * 0.075, text=str(y_max),
                                    border_line_color='black', border_line_alpha=0,
                                    background_fill_color='white', background_fill_alpha=1.0, text_align="center",
                                    text_line_height=1.5, text_font_size="10pt", text_color="darkred"

                                    ))

    load_markings = []
    for load_timing in cycles_loaded:
        load_markings.append(
            Span(location=load_timing[0], dimension="height", line_color="red", line_width=2, line_dash=[6, 3])
        )

    p.renderers.extend(load_markings)

    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def plot_aux_perf_progress(aux_perf: Dict[str, list], cycles_loaded, perf_name: str="Performance Metric"):
    """Plot auxiliary performance measure."""
    perf_name = " ".join([n.capitalize() for n in perf_name.split("_")])

    if isinstance(aux_perf, list):
        means = aux_perf
        stds = np.zeros_like(aux_perf)
    else:
        means, stds = aux_perf["mean"], aux_perf["std"]
    stds = np.array(stds)
    stds[stds == None] = 0
    stds = stds * 0.2
    means = np.array(means)
    means = mean_fill_nones(means)

    x = list(range(len(means)))
    df = pd.DataFrame(data=dict(x=x, y=means, lower=np.subtract(means, stds), upper=np.add(means, stds)))
    value_range = max(df["upper"]) - min(df["lower"])

    tooltip = HoverTool(
        tooltips=[("Cycle", "@x"),
                  ("Reward", "@y")],
        mode="vline"
    )

    range_max = max(df["upper"]) + value_range * 0.2
    p = figure(title=f"{perf_name} per Cycle",
               x_axis_label='Cycle',
               y_axis_label=perf_name,
               y_range=(min(df["lower"]), range_max),
               x_range=(0, max(x)),
               **plot_styling)

    p.add_tools(tooltip)
    p.add_tools(bokeh.models.BoxZoomTool())
    p.add_tools(bokeh.models.ResetTool())
    p.add_tools(bokeh.models.SaveTool())

    # ERROR BAND
    error_band = bokeh.models.Band(
        base="x", lower="lower", upper="upper",
        source=ColumnDataSource(df.reset_index()),
        fill_color=palette[0],
        fill_alpha=0.2,
        line_color=palette[0],
        line_alpha=0.4,
    )
    p.add_layout(error_band)
    p.renderers.extend([error_band])

    # REWARD LINE
    p.line(x, means, legend_label="Reward", line_width=2, color=palette[0])

    # MAX VALUE MARKING
    x_max = np.argmax(means)
    y_max = np.max(means)
    p.add_layout(bokeh.models.Arrow(end=bokeh.models.NormalHead(size=10,
                                                                line_color="darkred",
                                                                line_width=2,
                                                                fill_color="red"),
                                    line_color="darkred",
                                    line_width=2,
                                    x_start=x_max, y_start=y_max + value_range * 0.07,
                                    x_end=x_max, y_end=y_max))
    p.add_layout(bokeh.models.Label(x=x_max, y=y_max + value_range * 0.075, text=str(y_max),
                                    border_line_color='black', border_line_alpha=0,
                                    background_fill_color='white', background_fill_alpha=1.0, text_align="center",
                                    text_line_height=1.5, text_font_size="10pt", text_color="darkred"

                                    ))

    load_markings = []
    for load_timing in cycles_loaded:
        load_markings.append(
            Span(location=load_timing[0], dimension="height", line_color="red", line_width=2, line_dash=[6, 3])
        )

    p.renderers.extend(load_markings)

    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def compare_reward_progress(rewards: Dict[str, Dict[str, list]], reward_threshold=None):
    """Plot the execution times of a full cycle and optionally bot sub phases."""
    y_min = 0
    y_max = 0
    x_all = []

    p = figure(title="Average Rewards per Cycle",
               x_axis_label='Cycle',
               y_axis_label='Total Episode Return',
               **plot_styling)
    p.x_range.start = 0

    names_and_lines = []
    for i, (name, data) in enumerate(rewards.items()):
        means, stds = data["mean"], data["stdev"]
        stds = np.array(stds)
        stds[stds == None] = 0
        stds = stds * 0.2
        means = np.array(means)
        means = mean_fill_nones(means)

        y_min = min(min(means), y_min)
        y_max = max(max(means), y_max)

        x = list(range(len(means)))
        if len(x) > len(x_all):
            x_all = x

        df = pd.DataFrame(data=dict(x=x, y=means, lower=np.subtract(means, stds), upper=np.add(means, stds)))
        value_range = max(df["upper"]) - min(df["lower"])

        range_max = max(df["upper"]) + value_range * 0.1
        if reward_threshold is not None:
            range_max = max(range_max, reward_threshold * 1.1)

        # ERROR BAND
        error_band = bokeh.models.Band(
            base="x", lower="lower", upper="upper",
            source=ColumnDataSource(df.reset_index()),
            fill_color=palette[i % len(palette)],
            fill_alpha=0.2,
            line_color=palette[i % len(palette)],
            line_alpha=0.4,
        )
        p.add_layout(error_band)
        p.renderers.extend([error_band])

        # REWARD LINE
        reward_line = p.line(x, means, line_width=2, color=palette[i % len(palette)])
        names_and_lines.append((
            name, [reward_line]
        ))

    p.add_tools(bokeh.models.BoxZoomTool())
    p.add_tools(bokeh.models.ResetTool())
    p.add_tools(bokeh.models.SaveTool())

    # REWARD THRESHOLD
    if reward_threshold is not None:
        p.line(x_all, [reward_threshold for _ in x_all], line_color="green", line_width=2, line_alpha=0.7, line_dash="dashed",
               legend_label="Solution Threshold")

    p.y_range = bokeh.models.Range1d(y_min, y_max)

    legend = bokeh.models.Legend(items=names_and_lines, location="bottom_center")
    p.add_layout(legend, "below")

    style_plot(p)
    p.height = 750 + legend.glyph_height * len(rewards)

    return p


def make_grouped_reward_progress_figure(
        rewards: Dict[str, Tuple[int, Dict[str, list]]],
        reward_threshold,
        title=None,
        no_y_label=False
):
    y_min = 0
    y_max = 0 if reward_threshold is None else reward_threshold
    x_all = []

    p = figure(title="Average Rewards per Cycle" if title is None else title,
               x_axis_label='Cycle',
               y_axis_label='Total Episode Return' if not no_y_label else "",
               **plot_styling)
    p.x_range.start = 0

    by_groups = {}
    for name, (group, data) in rewards.items():
        if group not in by_groups.keys():
            by_groups[group] = []

        by_groups[group].append(data)

    i = 0
    names_and_lines = []
    tooltip_parts = []
    for group, datas in by_groups.items():
        means = np.mean([mean_fill_nones(d["mean"]) for d in datas], axis=0)
        stds = np.std([d["mean"] for d in datas], axis=0)

        y_min = min(min(means), y_min)
        y_max = max(max(means + stds), y_max)

        x = list(range(len(means)))
        if len(x) > len(x_all):
            x_all = x

        df = pd.DataFrame(data=dict(x=x, y=means, lower=np.subtract(means, stds), upper=np.add(means, stds)))
        value_range = max(df["upper"]) - min(df["lower"])

        tooltip_parts.append((f"Group {group}", "@y"))

        range_max = max(df["upper"]) + value_range * 0.1

        # ERROR BAND
        error_band = bokeh.models.Band(
            base="x", lower="lower", upper="upper",
            source=ColumnDataSource(df.reset_index()),
            fill_color=palette[i % len(palette)],
            fill_alpha=0.2,
            line_color=palette[i % len(palette)],
            line_alpha=0.4,
        )
        p.add_layout(error_band)
        p.renderers.extend([error_band])

        # REWARD LINE
        reward_line = p.line(x, means, line_width=2, color=palette[i % len(palette)])
        names_and_lines.append((
            str(group), [reward_line]
        ))

        i += 1

    tooltip_parts.append(("Cycle", "@x"))
    tooltip = HoverTool(
        tooltips=tooltip_parts,
        mode="vline"
    )

    p.add_tools(tooltip)

    p.add_tools(bokeh.models.BoxZoomTool())
    p.add_tools(bokeh.models.ResetTool())
    p.add_tools(bokeh.models.SaveTool())

    # REWARD THRESHOLD
    if reward_threshold is not None:
        p.line(x_all, [reward_threshold for _ in x_all], line_color="green", line_width=2, line_alpha=0.7,
               line_dash="dashed",
               legend_label="Solution Threshold")

    p.y_range = bokeh.models.Range1d(y_min, y_max)

    legend = bokeh.models.Legend(items=names_and_lines, location="bottom_center", orientation="horizontal")
    p.add_layout(legend, "below")

    style_plot(p)

    return p


def grouped_reward_progress(rewards: Dict[str, Tuple[int, Dict[str, list]]], reward_threshold=None):
    p = make_grouped_reward_progress_figure(rewards, reward_threshold)
    p.height = 750

    return p


def group_preview(groups: Dict[str, Dict[str, list]], alt_titles=None, reward_thresholds=None):
    figures = []
    for i, group in enumerate(groups.keys()):
        ids_with_rewards = groups[group]

        p = make_grouped_reward_progress_figure(
            {id: ("", progress["rewards"]) for id, progress in ids_with_rewards.items()},
            None if reward_thresholds is None else reward_thresholds[group],
            title=group if alt_titles is None else alt_titles[group],
            no_y_label=(i != 0)
        )

        p.legend.visible = False

        figures.append(p)

    gp = layouts.gridplot(figures, ncols=4, sizing_mode="scale_width")

    return gp


def plot_length_progress(lengths: Dict[str, list], cycles_loaded):
    """Plot development of the lengths of episodes throughout training."""
    means, stds = lengths["mean"], lengths["stdev"]
    stds = np.array(stds)
    stds[stds == None] = 0
    stds = stds * 0.2
    means = np.array(means)
    means[means == None] = 0

    x = list(range(len(means)))
    df = pd.DataFrame(data=dict(x=x, y=means, lower=np.subtract(means, stds), upper=np.add(means, stds)))
    value_range = max(df["upper"]) - min(df["lower"])

    tooltip = HoverTool(
        tooltips=[("Cycle", "@x"),
                  ("Length", "@y")],
        mode="vline"
    )

    p = figure(title="Average Length of Episodes",
               x_axis_label='Cycle',
               y_axis_label='Total Episode Return',
               y_range=(min(df["lower"]), max(df["upper"]) + value_range * 0.1),
               x_range=(0, max(x)),
               **plot_styling)

    p.add_tools(tooltip)
    p.add_tools(bokeh.models.BoxZoomTool())
    p.add_tools(bokeh.models.ResetTool())
    p.add_tools(bokeh.models.SaveTool())

    # ERROR BAND
    error_band = bokeh.models.Band(
        base="x", lower="lower", upper="upper",
        source=ColumnDataSource(df.reset_index()),
        fill_color=palette[1],
        fill_alpha=0.2,
        line_color=palette[1],
        line_alpha=0.4,
    )
    p.add_layout(error_band)
    p.renderers.extend([error_band])

    # REWARD LINE
    p.line(x, means, legend_label="Episode Length (time steps)", line_width=2, color=palette[1])

    # MAX VALUE MARKING
    x_max = np.argmax(means)
    y_max = np.max(means)
    p.add_layout(bokeh.models.Arrow(end=bokeh.models.NormalHead(size=15,
                                                                line_color="darkred",
                                                                line_width=2,
                                                                fill_color="red"),
                                    line_color="darkred",
                                    line_width=2,
                                    x_start=x_max, y_start=y_max + value_range * 0.1,
                                    x_end=x_max, y_end=y_max))
    p.add_layout(bokeh.models.Label(x=x_max, y=y_max + value_range * 0.1, text=str(y_max),
                                    border_line_color='black', border_line_alpha=1.0,
                                    background_fill_color='white', background_fill_alpha=1.0, text_align="center",
                                    text_line_height=1.5, text_font_size="10pt",

                                    ))

    load_markings = []
    for load_timing in cycles_loaded:
        load_markings.append(
            Span(location=load_timing[0], dimension="height", line_color="red", line_width=2, line_dash=[6, 3])
        )

    p.renderers.extend(load_markings)

    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def plot_distribution(metric: List[float], name="Metric", color=0) -> Tuple:
    hist, edges = np.histogram(metric, density=True, bins=50)

    p = figure(title=f"Distribution of {name}",
               x_axis_label=name,
               y_range=(0, np.max(hist)),
               **plot_styling)

    if len(metric) == 0:
        return p

    width = (max(metric) - min(metric)) / len(hist) * 0.9
    hist_glyph = p.rect(x=edges, y=hist / 2, height=hist, width=width, line_color="white", fill_color=palette[color])

    x, mean, sigma = np.linspace(min(metric), max(metric), 50), np.mean(metric), np.std(metric)

    tooltips = [("Reward", "@x")]
    try:
        pdf = stats.gaussian_kde(metric, bw_method=0.4).pdf(x)
        p.line(x, pdf, line_color=palette[color], line_width=4, alpha=0.7, )
        tooltips.append(("Density", "@y"))
    except np.linalg.LinAlgError as e:
        pass

    tooltip = HoverTool(
        tooltips=tooltips,
        mode="vline",
        renderers=[hist_glyph]
    )
    p.add_tools(tooltip)
    p.add_tools(bokeh.models.SaveTool())

    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def plot_loss(loss, rewards, name, color_id=0):
    """Plot a loss as it develops over cycles."""
    x = list(range(len(loss)))
    loss = np.array(loss, dtype=float)
    rewards = np.array(rewards, dtype=float)

    loss[np.isnan(loss)] = None

    p = figure(title=name,
               x_axis_label='Cycle',
               y_axis_label='Loss',
               y_range=(min(loss), max(loss)),
               **plot_styling)

    p.extra_y_ranges = {"Reward": Range1d(start=min(rewards), end=max(rewards))}
    p.add_layout(LinearAxis(y_range_name="Reward"), "right")

    p.line(x, rewards, legend_label="Reward", line_width=2, color="lightgrey", y_range_name="Reward")
    p.line(x, loss, legend_label=name, line_width=2, color=palette[color_id])

    p.add_tools(bokeh.models.SaveTool())


    p.legend.location = "bottom_right"
    style_plot(p)

    return p


def plot_preprocessor(preprocessor_data: Dict[str, List[Dict[str, float]]]):
    """Plot progression of preprocessor means."""
    x = list(range(len(preprocessor_data["mean"])))

    plots = []
    for i, sense in enumerate(preprocessor_data["mean"][0].keys()):
        p = figure(title=f"{sense.capitalize()}",
                   x_axis_label='Cycle',
                   y_axis_label='Running Mean',
                   **plot_styling)

        trace = [point[sense] for point in preprocessor_data["mean"]]
        if len(trace) > 1:
            trace = trace[1:]

            if i == 0:
                x = x[1:]

        p.line(x, trace, legend_label=str(sense), line_width=2, color=palette[i])

        p.add_tools(bokeh.models.SaveTool())

        p.legend.location = "bottom_right"
        style_plot(p)

        plots.append(p)

    p = bokeh.layouts.row(plots)
    p.sizing_mode = "stretch_width"
    return p
