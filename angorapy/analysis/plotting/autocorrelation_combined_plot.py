import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm

from angorapy.agent import PPOAgent


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

    data = np.diff(np.array(state_trace), 0)  # - np.mean(state_trace, axis=0)

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

        # acorr = sm.tsa.acf(timeseries, nlags=n_lags)
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
    n_samples = 2
    n_lags = 30

    violin_ax = plt.subplot2grid((3, 3), (0, 0), colspan=3, title="Pairwise Cross-correlations")
    acf_ax1 = plt.subplot2grid((3, 3), (1, 0), colspan=1, title="Pairwise Cross-correlations")
    acf_ax2 = plt.subplot2grid((3, 3), (1, 1), colspan=1, title="Autocorrelations")
    acf_ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, title="State Correlations")
    acf_ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1)
    acf_ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1)
    acf_ax6 = plt.subplot2grid((3, 3), (2, 2), colspan=1)

    agent = PPOAgent.from_agent_state(1673350499432390, from_iteration="best", path_modifier="../../../")
    autocorrelations, crosscorrelations, statewise_correlations = [], [], []
    for i in tqdm(range(n_samples)):
        data = get_timeseries_data(agent)
        data = np.array(data) - np.mean(data, axis=0)

        autocorrelations.append(calculate_timewise_autocorrelation(data, n_lags=n_lags + 1))
        crosscorrelations.append(calculate_timewise_crosscorrelations(data, n_lags=n_lags + 1))
        statewise_correlations.append(calculate_statewise_correlation(data, n_lags=n_lags + 1))

    autocorrelation = np.mean(np.stack(autocorrelations, 0), 0)
    crosscorrelation = np.mean(np.stack(crosscorrelations, 0), 0)
    statewise_correlation = np.mean(np.stack(statewise_correlations, 0), 0)

    lags = (np.ones_like(crosscorrelation) * np.expand_dims(np.arange(len(crosscorrelation)), 1)).flatten().astype(int)
    df = pd.DataFrame({"lag": lags, "correlation": np.array(crosscorrelation).flatten()})

    plt.gcf().set_size_inches(16, 8)
    sns.violinplot(data=df, x="lag", y="correlation", ax=violin_ax)
    # sns.swarmplot(data=df, x="lag", y="correlation", size=5 * 0.09, ax=violin_ax)
    # for i, l in enumerate(total_corr):
    #     plt.scatter(np.ones((len(l),)) * i, l)

    ax_i = 0
    acf_axes = [acf_ax1, acf_ax4, acf_ax2, acf_ax5]
    for include_cross_correlation, combine_by in itertools.product([True, False], ["mean", "max"]):
        print(f"plotting combined by {combine_by} with{'out' if not include_cross_correlation else ''} cross-correlation")

        a_or_c_corr = autocorrelation if not include_cross_correlation else crosscorrelation
        if combine_by == "mean":
            corrs_by_lags = np.mean(a_or_c_corr, axis=1)
        elif combine_by == "max":
            corrs_by_lags = np.max(a_or_c_corr, axis=1)

        acf_axes[ax_i].stem(range(n_lags + 1), corrs_by_lags)
        acf_axes[ax_i].set_xlabel("Lag")
        acf_axes[ax_i].set_ylabel(f"{combine_by.capitalize()} Absolute Correlation")
        acf_axes[ax_i].set_ylim(0, 1.1)

        ax_i += 1

    acf_ax3.stem(range(n_lags + 1), np.mean(statewise_correlation, axis=0))
    acf_ax3.set_xlabel("Lag")
    acf_ax3.set_ylabel("Mean Correlation")

    acf_ax6.stem(range(n_lags + 1), np.mean(np.abs(statewise_correlation), axis=0))
    acf_ax6.set_xlabel("Lag")
    acf_ax6.set_ylabel("Mean Absolute Correlation")
    acf_ax6.set_ylim(0, 1.1)

    plt.tight_layout()

    # plt.savefig("../../../docs/figures/combined-correlation-plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()
