import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import colors
from tqdm import tqdm

from angorapy import register_model
from angorapy.agent import PPOAgent
from dexterity.model import build_fpn_models, build_fpn_v2_models, build_fpn_v3_models

register_model(build_fpn_models)
register_model(build_fpn_v2_models)
register_model(build_fpn_v3_models)


def get_timeseries_data(agent, random_steps=True):
    env = agent.env
    if random_steps:
        agent.policy, agent.value, agent.joint = agent.build_models(
            agent.joint.get_weights(),
            batch_size=1,
            sequence_length=1
        )

    state_trace = []
    state, info = env.reset()
    for t in range(100):
        if not random_steps:
            action, _ = agent.distribution.act(*agent.policy(state.with_leading_dims(time=True).dict_as_tf()))
        else:
            action = env.action_space.sample()
        state, r, terminated, truncated, i = env.step(np.atleast_1d(action))
        state_trace.append(np.concatenate([
            state.dict()["proprioception"],
            state.dict()["vision"],
            # state.dict()["touch"] + np.random.normal(0, 0.01, state.dict()["touch"].shape),
            state.dict()["asymmetric"],
        ]))

        if terminated:
            state, info = env.reset()

    data = np.diff(np.array(state_trace), 0)

    return data


def calculate_timewise_autocorrelation(data, n_lags=30):
    """Calculate autocorrelations between timeseries at all lags up to n_lags."""
    total_corr = []
    for i_var in range(data.shape[1]):
        timeseries_i = data[:, i_var]
        total_corr.append(np.abs(sm.tsa.acf(timeseries_i, nlags=n_lags - 1)))

    return np.array(total_corr).T


def calculate_timewise_crosscorrelations(data, n_lags=30):
    """Calculate autocorrelations between timeseries at all lags up to n_lags."""
    total_corr = [[] for i in range(n_lags)]
    for i_var in range(data.shape[1]):
        timeseries_i = data[:, i_var]

        for j_var in range(i_var, data.shape[1]):
            timeseries_j = data[:, j_var]
            for lag in range(n_lags):
                corr = np.corrcoef(timeseries_i, np.roll(timeseries_j, lag))[0, 1]
                total_corr[lag].append(np.abs(corr))

    return total_corr


def calculate_statewise_correlation(data, n_lags=30):
    corr_coeffs = np.corrcoef(data)
    # corr_coeffs = ((data @ data.T) / (np.linalg.norm(data, axis=1) ** 2))
    lag_shifted_corr_coeffs = np.zeros_like(corr_coeffs)
    for row in range(n_lags):
        lag_shifted_corr_coeffs[row] = np.roll(corr_coeffs[row], -row)

    return lag_shifted_corr_coeffs[:n_lags, :n_lags]


if __name__ == '__main__':
    # EXPERIMENTAL PARAMETERS
    n_samples = 15
    n_lags = 30

    # PLOT PARAMETERS
    MARKER_SIZE = 4

    violin_ax = plt.subplot2grid((3, 3), (0, 0), colspan=3, title="Pairwise Cross-correlations")
    acf_ax1 = plt.subplot2grid((3, 3), (1, 0), colspan=1, title="Pairwise Cross-correlations")
    acf_ax2 = plt.subplot2grid((3, 3), (1, 1), colspan=1, title="Autocorrelations")
    acf_ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, title="State Correlations")
    acf_ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1)
    acf_ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1)
    acf_ax6 = plt.subplot2grid((3, 3), (2, 2), colspan=1)

    agent = PPOAgent.from_agent_state(1692396321151529, from_iteration="best", path_modifier="../../../")
    autocorrelations, crosscorrelations, statewise_correlations = [], [], []
    for i in tqdm(range(n_samples)):
        data = get_timeseries_data(agent)
        data = np.array(data) - np.mean(data, axis=0)

        autocorrelations.append(calculate_timewise_autocorrelation(data, n_lags=n_lags + 1))
        crosscorrelations.append(calculate_timewise_crosscorrelations(data, n_lags=n_lags + 1))
        statewise_correlations.append(calculate_statewise_correlation(data, n_lags=n_lags + 1))

    # stack samples
    sample_mean_autocorrelation = np.mean(np.stack(autocorrelations, 0), 0)
    sample_mean_crosscorrelation = np.mean(np.stack(crosscorrelations, 0), 0)
    sample_mean_statewise_correlation = np.mean(np.stack(statewise_correlations, 0), 0)

    # make crosscorrelation violin plot
    lags = (np.ones_like(sample_mean_crosscorrelation) * np.expand_dims(
        np.arange(sample_mean_crosscorrelation.shape[0]), [1])).flatten().astype(int)
    df = pd.DataFrame({"Lag": lags, "Correlation": np.array(sample_mean_crosscorrelation).flatten()})

    plt.gcf().set_size_inches(16, 8)

    # violoinplot with all violins filled with color lightblue
    sns.violinplot(
        data=df,
        x="Lag",
        y="Correlation",
        ax=violin_ax,
        linecolor="tab:blue",
        color="skyblue",
        saturation=1,
        inner_kws=dict(
            box_width=3,
            whis_width=0,
            markerfacecolor="black",
            markeredgecolor="black",
        )
    )

    # make auto and crosscorrelation plots
    ax_i = 0
    acf_axes = [acf_ax1, acf_ax4, acf_ax2, acf_ax5]
    for a_or_c_corr, combine_by in itertools.product([sample_mean_crosscorrelation, sample_mean_autocorrelation],
                                                     ["mean", "max"]):
        if combine_by == "mean":
            variable_mean_correlations = np.mean(a_or_c_corr, axis=-1)

            variable_se_correlations = (
                    np.std(a_or_c_corr, axis=-1)
                    / np.sqrt(a_or_c_corr.shape[-1])
            )

            acf_axes[ax_i].fill_between(
                range(n_lags + 1),
                variable_mean_correlations - variable_se_correlations,
                variable_mean_correlations + variable_se_correlations,
                color="skyblue"
            )
        elif combine_by == "max":
            variable_mean_correlations = np.max(a_or_c_corr, axis=-1)
        else:
            raise ValueError(f"combine_by must be 'mean' or 'max', not {combine_by}")

        markerline, stemlines, baseline = acf_axes[ax_i].stem(range(n_lags + 1), variable_mean_correlations)
        markerline.set_markersize(MARKER_SIZE)

        acf_axes[ax_i].vlines(
            16,
            ymin=0,
            ymax=1,
            linestyles="dashed",
            color="red"
        )

        acf_axes[ax_i].set_xlabel("Lag")
        acf_axes[ax_i].set_ylabel(f"{combine_by.capitalize()} Absolute Correlation")
        acf_axes[ax_i].set_ylim(0, 1.1)

        ax_i += 1

    # make statewise correlation plot
    variable_mean_correlations = np.mean(sample_mean_statewise_correlation, axis=-2)
    variable_ci_statewise_correlation = (
            np.std(sample_mean_statewise_correlation, axis=-2)
            / np.sqrt(sample_mean_statewise_correlation.shape[-2])
    )

    markerline, stemlines, baseline = acf_ax6.stem(range(n_lags + 1), variable_mean_correlations)
    markerline.set_markersize(MARKER_SIZE)
    acf_ax6.fill_between(
        range(n_lags + 1),
        variable_mean_correlations - variable_ci_statewise_correlation,
        variable_mean_correlations + variable_ci_statewise_correlation,
        color="skyblue"
    )
    acf_ax6.set_xlabel("Lag")
    acf_ax6.set_ylabel("Mean Correlation")

    acf_ax6.vlines(
        16,
        ymin=0,
        ymax=1,
        linestyles="dashed",
        color="red"
    )

    # make statewise correlation plot
    variable_mean_correlations = np.mean(np.abs(sample_mean_statewise_correlation), axis=-2)
    variable_ci_statewise_correlation = (
            np.std(np.abs(sample_mean_statewise_correlation), axis=-2)
            / np.sqrt(sample_mean_statewise_correlation.shape[-2])
    )

    markerline, stemlines, baseline = acf_ax3.stem(range(n_lags + 1), variable_mean_correlations)
    markerline.set_markersize(MARKER_SIZE)
    acf_ax3.fill_between(
        range(n_lags + 1),
        variable_mean_correlations - variable_ci_statewise_correlation,
        variable_mean_correlations + variable_ci_statewise_correlation,
        color="skyblue"
    )
    acf_ax3.set_xlabel("Lag")
    acf_ax3.set_ylabel("Mean Absolute Correlation")
    acf_ax3.set_ylim(0, 1.1)

    acf_ax3.vlines(
        16,
        ymin=0,
        ymax=1,
        linestyles="dashed",
        color="red"
    )

    plt.tight_layout()

    plt.savefig("../../../docs/figures/combined-correlation-plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()
