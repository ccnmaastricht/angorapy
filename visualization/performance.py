#!/usr/bin/env python
"""Visualization methods for performance plotting."""
from typing import List

import matplotlib.pyplot as plt


def plot_performance_over_episodes(episodes: List[List[float]], performances: List[List[float]], models: List[str]):

    for i, model in enumerate(models):
        plt.plot(episodes[i], performances[i], label=model)
        plt.xlabel("Episodes Seen")
        plt.ylabel("Reward")

        plt.legend()
        plt.show()
