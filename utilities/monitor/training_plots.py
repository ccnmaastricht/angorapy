from typing import List

from bokeh import embed, colors
from bokeh.plotting import figure

from common.const import COLORMAP

palette = [colors.RGB(*[int(c * 255) for c in color]) for color in COLORMAP]
darker_palette = [c.darken(0.3) for c in palette]

tooltip_css = """
tooltip {
    background-color: #212121;
    color: white;
    padding: 5px;
    border-radius: 10px;
    margin-left: 10px;
}
"""

plot_styling = dict(
    plot_height=500,
    sizing_mode="stretch_width",
    toolbar_location=None,
)


def style_plot(p):
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.outline_line_color = None

    p.axis.axis_label_text_font = "times"
    p.axis.axis_label_text_font_size = "12pt"
    p.axis.axis_label_text_font_style = "bold"

    p.legend.label_text_font = "times"
    p.legend.label_text_font_size = "12pt"
    p.legend.label_text_font_style = "normal"

    p.title.align = "center"
    p.title.text_font_size = "14pt"
    p.title.text_font = "Fira Sans"


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
