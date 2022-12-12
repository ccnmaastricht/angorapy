#!/usr/bin/env python
"""Custom data types for simplifying data communication."""
import itertools
import statistics
from collections import namedtuple
from typing import List, Union

import numpy as np
from mpi4py import MPI

StatBundle = namedtuple("StatBundle", ["numb_completed_episodes", "numb_processed_frames",
                                       "episode_rewards", "episode_lengths", "tbptt_underflow",
                                       "per_receptor_mean", "auxiliary_performances"])


def condense_stats(stat_bundles: List[StatBundle]) -> Union[StatBundle, None]:
    """Infer a single StatBundle from a list of StatBundles."""
    return StatBundle(
        numb_completed_episodes=sum([s.numb_completed_episodes for s in stat_bundles]),
        numb_processed_frames=sum([s.numb_processed_frames for s in stat_bundles]),
        episode_rewards=list(itertools.chain(*[s.episode_rewards for s in stat_bundles])),
        episode_lengths=list(itertools.chain(*[s.episode_lengths for s in stat_bundles])),
        tbptt_underflow=round(statistics.mean(map(lambda x: x.tbptt_underflow, stat_bundles)), 2) if (
                stat_bundles[0].tbptt_underflow is not None) else None,
        per_receptor_mean={
            sense: np.mean([s.per_receptor_mean[sense] for s in stat_bundles], axis=0)
            for sense in stat_bundles[0].per_receptor_mean.keys()
        },
        auxiliary_performances={
            aux: list(itertools.chain(*[s.auxiliary_performances[aux] for s in stat_bundles]))
            for aux in stat_bundles[0].auxiliary_performances.keys()
        }
    )


def mpi_condense_stats(stat_bundles: List[StatBundle]) -> Union[StatBundle, None]:
    """Pull together the StatBundle lists from the buffer on all workers and infer a single StatBundle."""
    stat_bundles = MPI.COMM_WORLD.gather(stat_bundles, root=0)

    if MPI.COMM_WORLD.Get_rank() == 0:
        stat_bundles = list(itertools.chain(*stat_bundles))
        return condense_stats(stat_bundles)
    else:
        return None
