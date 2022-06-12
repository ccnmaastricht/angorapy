from gym.spaces import Box, Discrete, MultiDiscrete

from angorapy.common.policies import get_distribution_by_short_name


def autoselect_distribution(env):
    """For a given environment, select a compatible default distribution."""
    if isinstance(env.action_space, Box):
        distribution = "gaussian"
    elif isinstance(env.action_space, Discrete):
        distribution = "categorical"
    elif isinstance(env.action_space, MultiDiscrete):
        distribution = "multi-categorical"
    else:
        raise NotImplementedError("No appropriate distribution found for environment.")

    return get_distribution_by_short_name(distribution)(env)
