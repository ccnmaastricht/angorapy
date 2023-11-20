import os
import sys
from typing import List

import numpy as np
import tensorflow as tf
import tqdm
from angorapy.utilities.core import flatten
from angorapy.utilities.core import stack_dicts
from matplotlib import pyplot as plt

from angorapy.analysis.investigators import base_investigator
from angorapy.common.policies import BasePolicyDistribution
from angorapy.tasks.wrappers import TaskWrapper


class ActivityAutocorrelation(base_investigator.Investigator):
    """Investigate autocorrelation of the activity in different regions of the network as a measure of how long
    information is maintained over time."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)
        self._data = None
        self.prepared = False

        self.n_states = np.nan

    def prepare(self, env: TaskWrapper, layers: List[str], n_states=1000, n_repeats=1, verbose=True):
        """Prepare samples to predict from."""
        self.network.reset_states()
        state, info = env.reset(return_info=True)

        repeated_collections = []
        for i in range(n_repeats):
            activation_collection = []
            for _ in tqdm.tqdm(range(n_states), disable=not verbose):
                prepared_state = state.with_leading_dims(time=self.is_recurrent).dict()

                # make step and record activities
                activations = self.get_layer_activations(layers, prepared_state)
                probabilities = flatten(activations["output"])
                activations.pop("output")
                activation_collection.append(activations)

                action, _ = self.distribution.act(*probabilities)

                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    state, info = env.reset(return_info=True)
                    self.network.reset_states()
                else:
                    state = observation

            repeated_collections.append(activation_collection)

        self._data = stack_dicts([stack_dicts(ac) for ac in repeated_collections])
        self.prepared = True

    def fit(self, n_lags=30):
        """Measure the predictability of target_information based on the information in source_layer's activation."""
        assert self.prepared, "Need to prepare before investigating."

        # calculate autocorrelation for each layer
        autocorrelation = {}
        for layer in self._data.keys():
            layer_autocorrelations = []
            for repeat in range(self._data[layer].shape[0]):
                layer_autocorrelations.append(self.calculate_statewise_correlation(
                    np.squeeze(self._data[layer][repeat])),
                    n_lags=n_lags
                )

            autocorrelation[layer] = np.mean(layer_autocorrelations, axis=0)

        return autocorrelation

    def calculate_statewise_correlation(self, data, n_lags=30):
        corr_coeffs = np.corrcoef(data)
        lag_shifted_corr_coeffs = np.zeros_like(corr_coeffs)
        for row in range(n_lags):
            lag_shifted_corr_coeffs[row] = np.roll(corr_coeffs[row], -row)

        return lag_shifted_corr_coeffs[:n_lags, :n_lags]


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../dexterity-fpn')))

    from angorapy.agent.ppo_agent import PPOAgent
    from dexterity.model import build_fpn_models, build_fpn_v2_models, build_fpn_v3_models
    from angorapy.common.const import QUALITATIVE_COLOR_PALETTE as COLORS
    import matplotlib

    font = {'family': 'Times',
            'weight': 'normal',
            'size': 12}

    matplotlib.rc('font', **font)

    register_model(build_fpn_models)
    register_model(build_fpn_v2_models)
    register_model(build_fpn_v3_models)

    agent = PPOAgent.from_agent_state(1692396321151529, "best", path_modifier="../../../")
    investigator = ActivityAutocorrelation.from_agent(agent)
    env = agent.env

    investigator.prepare(env,
                         layers=[
                             "SSC_internal",
                             "LPFC_internal",
                             "MCC_internal",
                             "IPL_internal",
                             "SPL_internal",
                             "IPS_internal",
                             "pmc_recurrent_layer",
                             "m1_internal",
                         ], n_states=1000, n_repeats=10, verbose=True)

    statewise_correlation = investigator.fit()

    n_lags = 30

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), sharey="row")

    for i, layer in enumerate(["LPFC_internal", "pmc_recurrent_layer", "m1_internal"]):
        axs[i].set_title(layer.split("_")[0].upper())
        markerline, stemlines, baseline = axs[i].stem(range(n_lags), np.mean(statewise_correlation[layer], axis=0))

        # set color of marker and stem lines
        plt.setp(markerline, 'markerfacecolor', COLORS[i])
        plt.setp(markerline, 'markeredgecolor', COLORS[i])
        plt.setp(stemlines, 'color', COLORS[i])

        # set size of marker
        plt.setp(markerline, 'markersize', 4)

        # set linewidth of stem lines
        plt.setp(stemlines, 'linewidth', 2)

        axs[i].set_xlabel("Lag")

        if i == 0:
            axs[i].set_ylabel("Mean Correlation")

        # hide y ticks on all but first plot of shared y axis
        if i > 0:
            axs[i].tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.tight_layout()

    plt.savefig("activity_autocorrelation.pdf", format="pdf", bbox_inches="tight")
    plt.show()
