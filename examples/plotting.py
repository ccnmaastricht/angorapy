import json

import angorapy as ang
import angorapy.environments
import bokeh
import numpy as np
from bokeh.io import output_notebook, show, export_png, export_svg

from angorapy.analysis.investigation import Investigator
from angorapy.common.const import PATH_TO_EXPERIMENTS
from angorapy.utilities.monitor import training_plots as plots

agent = ang.agent.PPOAgent.from_agent_state(1653053413, from_iteration="best", path_modifier="../")
investigator = Investigator.from_agent(agent)

with open(f"../{PATH_TO_EXPERIMENTS}/{agent.agent_id}/progress.json", "r") as f:
    progress_data = json.load(f)

reward_plot = plots.plot_reward_progress(progress_data["rewards"], [])
reward_plot.background_fill_alpha = 0
reward_plot.outline_line_alpha = 0
reward_plot.title = ""

reward_plot.output_backend = "svg"

export_svg(reward_plot, filename="plot.svg")

