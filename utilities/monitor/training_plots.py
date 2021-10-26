from typing import List, Dict


from bokeh import embed
from bokeh.models import Span, Range1d, LinearAxis
from bokeh.plotting import figure

from utilities.monitor.plotting_base import palette, plot_styling, style_plot


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

    return embed.components(p)


def plot_execution_times(cycle_timings, optimization_timings=None, gathering_timings=None):
    """Plot the execution times of a full cycle and optionally bot sub phases."""
    x = list(range(len(cycle_timings)))

    all_times = (cycle_timings
                 + (optimization_timings if optimization_timings is not None else [])
                 + (gathering_timings if gathering_timings is not None else []))

    p = figure(title="Execution Times",
               x_axis_label='Cycle',
               y_axis_label='Seconds',
               y_range=(min(all_times), max(all_times)),
               **plot_styling)

    p.line(x, cycle_timings, legend_label="Full Cycle", line_width=2, color=palette[0])
    if optimization_timings is not None:
        p.line(x, optimization_timings, legend_label="Optimization Phase", line_width=2, color=palette[1])
    if gathering_timings is not None:
        p.line(x, gathering_timings, legend_label="Gathering Phase", line_width=2, color=palette[2])

    p.legend.location = "bottom_right"
    style_plot(p)

    return embed.components(p)


def plot_reward_progress(rewards, cycles_loaded):
    """Plot the execution times of a full cycle and optionally bot sub phases."""
    x = list(range(len(rewards)))

    p = figure(title="Average Rewards per Cycle",
               x_axis_label='Cycle',
               y_axis_label='Total Episode Return',
               y_range=(min(rewards), max(rewards)),
               **plot_styling)

    p.line(x, rewards, legend_label="Reward", line_width=2, color=palette[0])

    load_markings = []
    for load_timing in cycles_loaded:
        load_markings.append(
            Span(location=load_timing[0], dimension="height", line_color="red", line_width=2, line_dash=[6, 3])
        )

    p.renderers.extend(load_markings)

    p.legend.location = "bottom_right"
    style_plot(p)

    return embed.components(p)


def plot_loss(loss, rewards, name, color_id=0):
    """Plot a loss as it develops over cycles."""
    x = list(range(len(loss)))

    p = figure(title=name,
               x_axis_label='Cycle',
               y_axis_label='Loss',
               y_range=(min(loss), max(loss)),
               **plot_styling)

    p.extra_y_ranges = {"Reward": Range1d(start=min(rewards), end=max(rewards))}
    p.add_layout(LinearAxis(y_range_name="Reward"), "right")

    p.line(x, rewards, legend_label="Reward", line_width=2, color="lightgrey", y_range_name="Reward")
    p.line(x, loss, legend_label=name, line_width=2, color=palette[color_id])

    p.legend.location = "bottom_right"
    style_plot(p)

    return embed.components(p)


def plot_preprocessor(preprocessor_data: Dict[str, List[Dict[str, float]]]):
    """Plot progression of preprocessor means."""
    x = list(range(len(preprocessor_data["mean"])))

    p = figure(title="Preprocessor Running Mean",
               x_axis_label='Cycle',
               y_axis_label='Running Mean',
               # y_range=(min(all_times), max(all_times)),
               **plot_styling)

    for i, sense in enumerate(preprocessor_data["mean"][0].keys()):
        trace = [point[sense] for point in preprocessor_data["mean"]]
        if len(trace) > 1:
            trace = trace[1:]

        p.line(x, trace, legend_label=sense, line_width=2, color=palette[i])

    p.legend.location = "bottom_right"
    style_plot(p)

    return embed.components(p)