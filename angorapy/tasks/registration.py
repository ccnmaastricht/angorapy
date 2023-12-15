from typing import Union

import gymnasium as gym

from angorapy.common.postprocessors import StateNormalizer, RewardNormalizer, BasePostProcessor
from angorapy.tasks.wrappers import TaskWrapper, PostProcessingWrapper
from angorapy.utilities.core import env_extract_dims


def make_task(env_name,
              reward_config: Union[str, dict] = None,
              reward_function: Union[str, dict] = None,
              postprocessors=None,
              **kwargs):
    """Make environment, including a possible reward config and postprocessors.

    If postprocessors is None, the default postprocessors are used, i.e. StateNormalizationPostProcessor and
    RewardNormalizationPostProcessor.

    Args:
        env_name (str): The registered name of the environment to make.
        reward_config (Union[str, dict]): The reward config to use.
        reward_function (Union[str, dict]): The reward function to use.
        postprocessors (Optional[List[BasePostProcessor]]): The postprocessors to use. Can be a list of instances of
            BasePostProcessor or a list of BasePostProcessor subclasses for which instances will be newly created.

    Returns:
        TaskWrapper: A wrapped instance of the task.
    """
    if postprocessors is None:
        postprocessors = [StateNormalizer, RewardNormalizer]

    base_env = gym.make(env_name, **kwargs)
    state_dim, n_actions = env_extract_dims(base_env)

    if postprocessors is None:
        postprocessors = []
    elif all(isinstance(t, BasePostProcessor) for t in postprocessors):
        postprocessors = postprocessors
    elif all(callable(t) for t in postprocessors):
        postprocessors = [t(env_name, state_dim, n_actions) for t in postprocessors]

    if len(postprocessors) > 0:
        env = PostProcessingWrapper(base_env, postprocessors=postprocessors)
    else:
        env = TaskWrapper(base_env)

    if reward_function is not None and hasattr(env, "reward_function"):
        env.set_reward_function(reward_function)

    if reward_config is not None and hasattr(env, "reward_config"):
        env.set_reward_config(reward_config)

    return env


make_env = make_task  # alias for backwards compatibility
