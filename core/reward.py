import gym
import numpy as np

from gym.envs.robotics.hand.reach import FINGERTIP_SITE_NAMES
from environments.shadowhand import get_fingertip_distance

FORCE_MULTIPLIER = 0.05
# ratio between the size of penalty zone of auxiliary fingertips and reard zone of target fingertip
AUXILIARY_DISTANCE_THRESHOLD_RATIO = 10


def calculate_force_penalty(simulator):
    """Calculate a penalty for applying force. Sum of squares of forces, ignoring the wrist."""
    return (simulator.data.actuator_force[2:] ** 2).sum() * FORCE_MULTIPLIER


def free_reach(env, info, punish_used_force=False):
    """Reward the relative join of the thumb's and a target finger's fingertip while punishing the closeness of other
    fingertips."""
    thumb_position = env._get_thumb_position()

    reward = (- get_fingertip_distance(thumb_position, env._get_target_finger_position())
              + info["is_success"] * env.success_multiplier)

    if punish_used_force:
        reward -= calculate_force_penalty(env.sim)

    for i, fname in enumerate(FINGERTIP_SITE_NAMES):
        if fname == env.thumb_name or i == np.where(env.goal == 1)[0].item():
            continue

        fingertip_distance = get_fingertip_distance(thumb_position, env._get_finger_position(fname))
        if not fingertip_distance > env.distance_threshold * AUXILIARY_DISTANCE_THRESHOLD_RATIO:
            reward += 0.2 * fingertip_distance

    return reward


def free_reach_old(env, info, punish_used_force=True):
    """Reward the relative join of the thumb's and a target finger's fingertip while punishing the closeness of other
    fingertips."""
    reward = (- get_fingertip_distance(env._get_thumb_position(), env._get_target_finger_position())
              + info["is_success"] * env.success_multiplier)

    if punish_used_force:
        reward -= calculate_force_penalty(env.sim)

    for i, fname in enumerate(FINGERTIP_SITE_NAMES):
        if fname == env.thumb_name or i == np.where(env.goal == 1)[0].item():
            continue

        fingertip_distance = get_fingertip_distance(env._get_thumb_position(), env._get_finger_position(fname))
        reward += 0.2 * fingertip_distance

    return reward


def sequential_free_reach(env, info: dict):
    """Reward following a sequence of reach movements."""

    # retrieve the id of the current goal finger
    current_goal_finger_id = env.goal_sequence[env.current_sequence_position]
    current_goal_finger_name = FINGERTIP_SITE_NAMES[current_goal_finger_id]

    # retrieve the id of the previous goal finger
    last_goal_finger_id = env.goal_sequence[0]
    if env.current_sequence_position > 0:
        last_goal_finger_id = env.goal_sequence[env.current_sequence_position - 1]

    # calculate distance between thumb and current goal finger and determine reward accordingly
    d = get_fingertip_distance(env._get_thumb_position(), env._get_finger_position(current_goal_finger_name))
    reward = -d + info["is_success"] * env.success_multiplier

    # incentivise distance to non target fingers
    for i, fname in enumerate(FINGERTIP_SITE_NAMES):
        # do not reward distance to env, thumb and last target (to give time to move away last target)
        if fname == env.thumb_name or i == current_goal_finger_id or i == last_goal_finger_id:
            continue

        reward += 0.2 * get_fingertip_distance(env._get_thumb_position(), env._get_finger_position(fname))

    # add constant punishment to incentivize progress; harm is scaled by position in sequence such that there is no
    # incentive to not reach a sequence goal because it would again add a lot of distance punishment
    reward -= (0.1 * (len(env.goal_sequence) - env.current_sequence_position))

    return reward
