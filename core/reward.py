from typing import List

import numpy as np
from gym.envs.robotics.hand.reach import FINGERTIP_SITE_NAMES

from environments.shadowhand import get_fingertip_distance


# HELPER FUNCTIONS

def calculate_force_penalty(simulator) -> float:
    """Calculate a penalty for applying force. Sum of squares of forces, ignoring the wrist."""
    return (simulator.data.actuator_force[2:] ** 2).sum()


def calculate_auxiliary_finger_penalty(environment, exclude: List[int] = None) -> float:
    """Calculate a penalty for auxiliary (non-target) fingers being in a zone around the thumb."""
    if exclude is None:
        exclude = []

    penalty = 0
    for i, fname in enumerate(FINGERTIP_SITE_NAMES):
        if fname == environment.thumb_name or i == np.where(environment.goal == 1)[0].item() or i in exclude:
            continue

        fingertip_distance = get_fingertip_distance(environment.get_thumb_position(),
                                                    environment.get_finger_position(fname))

        # cap reward at auxiliary zone, scale to focus on the target finger's movement
        penalty += (min(fingertip_distance, environment.reward_config["AUXILIARY_ZONE_RADIUS"])
                    * environment.reward_config["AUXILIARY_PENALTY_MULTIPLIER"])

    return - penalty


# REWARD FUNCTIONS

def reach(env, achieved_goal, goal, info: dict):
    """Simple reward function for reaching tasks, combining distance, force and success."""
    return (- get_fingertip_distance(achieved_goal, goal)
            + info["is_success"] * env.reward_config["SUCCESS_REWARD_MULTIPLIER"]
            - calculate_force_penalty(env.sim) * env.reward_config["FORCE_MULTIPLIER"])


def free_reach(env, achieved_goal, goal, info: dict):
    """Reward the relative join of the thumb's and a target's fingertip while punishing close other fingertips."""

    thumb_position = env.get_thumb_position()

    reward = (- get_fingertip_distance(thumb_position, env.get_target_finger_position())
              + info["is_success"] * env.reward_config["SUCCESS_MULTIPLIER"]
              - calculate_force_penalty(env.sim) * env.reward_config["FORCE_MULTIPLIER"])

    reward -= calculate_auxiliary_finger_penalty(env)

    return reward


def free_reach_positive_reinforcement(env, achieved_goal, goal, info: dict):
    """Reward progress towards the goal position while punishing other fingers for interfering."""
    # positive reinforcement
    progress_reward = (
            get_fingertip_distance(env.get_thumbs_previous_position(), env.get_target_fingers_previous_position())
            - get_fingertip_distance(env.get_thumb_position(), env.get_target_finger_position()))
    success_reward = info["is_success"] * env.reward_config["SUCCESS_MULTIPLIER"]
    reinforcement_reward = progress_reward + success_reward

    # positive punishment
    penalty = (calculate_force_penalty(env.sim) * env.reward_config["FORCE_MULTIPLIER"]
               + calculate_auxiliary_finger_penalty(env))

    # total reward
    return reinforcement_reward - penalty


def sequential_free_reach(env, achieved_goal, goal, info: dict):
    """Reward following a sequence of reach movements."""

    # reinforcement
    progress_reward = (
            get_fingertip_distance(env.get_thumbs_previous_position(), env.get_target_fingers_previous_position())
            - get_fingertip_distance(env.get_thumb_position(), env.get_target_finger_position()))
    success_reward = info["is_success"] * env.reward_config["SUCCESS_MULTIPLIER"]
    reinforcement_reward = progress_reward + success_reward

    # punishment
    exclusions = []
    if len(env.goal_sequence) > 1:
        exclusions = [env.current_target_finger]

    punishment_reward = (- calculate_force_penalty(env.sim) * env.reward_config["FORCE_MULTIPLIER"]
                         - calculate_auxiliary_finger_penalty(env, exclude=exclusions))

    return reinforcement_reward + punishment_reward
