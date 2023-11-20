from typing import Union

import gymnasium as gym

from angorapy.common.postprocessors import StateNormalizer, RewardNormalizer, BasePostProcessor
from angorapy.tasks.wrappers import TaskWrapper, TransformationWrapper
from angorapy.utilities.core import env_extract_dims


def make_task(env_name,
              reward_config: Union[str, dict] = None,
              reward_function: Union[str, dict] = None,
              transformers=None,
              **kwargs):
    """Make environment, including a possible reward config and transformers.

    If transformers is None, the default transformers are used, i.e. StateNormalizationTransformer and
    RewardNormalizationTransformer.

    Args:
        env_name (str): The registered name of the environment to make.
        reward_config (Union[str, dict]): The reward config to use.
        reward_function (Union[str, dict]): The reward function to use.
        transformers (Optional[List[BaseTransformer]]): The transformers to use.

    Returns:
        TaskWrapper: A wrapped instance of the task.
    """
    if transformers is None:
        transformers = [StateNormalizer, RewardNormalizer]

    base_env = gym.make(env_name, **kwargs)
    state_dim, n_actions = env_extract_dims(base_env)

    if transformers is None:
        transformers = []
    elif all(isinstance(t, BasePostProcessor) for t in transformers):
        transformers = transformers
    elif all(callable(t) for t in transformers):
        transformers = [t(env_name, state_dim, n_actions) for t in transformers]

    if len(transformers) > 0:
        env = TransformationWrapper(base_env, transformers=transformers)
    else:
        env = TaskWrapper(base_env)

    if reward_function is not None and hasattr(env, "reward_function"):
        env.set_reward_function(reward_function)

    if reward_config is not None and hasattr(env, "reward_config"):
        env.set_reward_config(reward_config)

    return env


make_env = make_task  # alias for backwards compatibility
