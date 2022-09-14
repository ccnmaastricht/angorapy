from typing import List

import numpy as np
import tqdm
from angorapy.environments.hand.shadowhand import FINGERTIP_SITE_NAMES

from angorapy.environments.hand.shadowhand import get_fingertip_distance


# HELPER FUNCTIONS

def calculate_force_penalty(data) -> float:
    """Calculate a penalty for applying force. Sum of squares of forces, ignoring the wrist."""
    return (data.actuator_force[2:] ** 2).sum()


def calculate_tendon_stress_penalty(data) -> float:
    """Calculate a penalty for applying force. Sum of squares of forces, ignoring the wrist. Scaled up by a constant
    to match force penalty in average scale."""
    return ((data.ten_length ** 2) * 116).sum()


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

        # cap reward at auxiliary zone, scale to focus on the target finger'serialization movement
        penalty += ((min(fingertip_distance, environment.reward_config["AUXILIARY_ZONE_RADIUS"])
                     - environment.reward_config[
                         "AUXILIARY_ZONE_RADIUS"])  # base reward is 0, being in the zone is punished
                    * environment.reward_config["AUXILIARY_PENALTY_MULTIPLIER"])

    return - penalty


# REWARD FUNCTIONS

def reach(env, achieved_goal, goal, info: dict):
    """Simple reward function for reaching tasks, combining distance, force and success."""
    return (- get_fingertip_distance(achieved_goal, goal)
            + info["is_success"] * env.reward_config["SUCCESS_BONUS"]
            - calculate_force_penalty(env.data) * env.reward_config["FORCE_MULTIPLIER"])


def free_reach(env, achieved_goal, goal, info: dict):
    """Reward the relative join of the thumb'serialization and a target'serialization fingertip while punishing close
    other fingertips."""

    thumb_position = env.get_thumb_position()

    reward = (- get_fingertip_distance(thumb_position, env.get_target_finger_position())
              + info["is_success"] * env.reward_config["SUCCESS_BONUS"]
              - calculate_force_penalty(env.data) * env.reward_config["FORCE_MULTIPLIER"])

    reward -= calculate_auxiliary_finger_penalty(env)

    return reward


def free_reach_positive_reinforcement(env, achieved_goal, goal, info: dict):
    """Reward progress towards the goal position while punishing other fingers for interfering."""
    # positive reinforcement
    progress_reward = (
            get_fingertip_distance(env.get_thumbs_previous_position(), env.get_target_fingers_previous_position())
            - get_fingertip_distance(env.get_thumb_position(), env.get_target_finger_position()))
    success_reward = info["is_success"] * env.reward_config["SUCCESS_BONUS"]
    reinforcement_reward = progress_reward + success_reward

    # positive punishment
    penalty = (calculate_force_penalty(env.data) * env.reward_config["FORCE_MULTIPLIER"]
               + calculate_auxiliary_finger_penalty(env))

    # total reward
    return reinforcement_reward - penalty


# SEQUENTIAL REACHING

def sequential_reach(env, achieved_goal, goal, info: dict):  # TODO adjust for sequence?
    """Reward following a sequence of finger configurations."""
    reward = (- get_fingertip_distance(achieved_goal, goal)
              + info["is_success"] * env.reward_config["SUCCESS_BONUS"]
              - calculate_force_penalty(env.data) * env.reward_config["FORCE_MULTIPLIER"])

    return reward


def sequential_free_reach(env, achieved_goal, goal, info: dict):
    """Reward following a sequence of reach movements."""

    # reinforcement
    progress_reward = (
            get_fingertip_distance(env.get_thumbs_previous_position(), env.get_target_fingers_previous_position())
            - get_fingertip_distance(env.get_thumb_position(), env.get_target_finger_position()))
    success_reward = info["is_success"] * env.reward_config["SUCCESS_BONUS"]
    reinforcement_reward = progress_reward + success_reward

    # punishment
    exclusions = []
    if len(env.goal_sequence) > 1:
        exclusions = [env.current_target_finger]

    punishment_reward = (- calculate_force_penalty(env.data) * env.reward_config["FORCE_MULTIPLIER"]
                         - calculate_auxiliary_finger_penalty(env, exclude=exclusions))

    return reinforcement_reward + punishment_reward


# MANIPULATE

def manipulate(env, achieved_goal, goal, info: dict):
    """Reward function fo object manipulation tasks."""
    success = env._is_success(achieved_goal, goal).astype(np.float32)
    d_pos, d_rot = env._goal_distance(achieved_goal, goal)
    d_progress = env._goal_progress()

    reward = (+ d_progress  # convergence to goal reward
              + success * env.reward_config["SUCCESS_BONUS"]  # reward for finishing
              - env._is_dropped() * env.reward_config["DROPPING_PENALTY"])  # dropping penalty

    reward -= calculate_force_penalty(env.data) * env.reward_config["FORCE_MULTIPLIER"]
    reward -= calculate_tendon_stress_penalty(env.data) * env.reward_config["TENDON_STRESS_MULTIPLIER"]

    return reward


if __name__ == '__main__':
    from environments import *

    env = gym.make("HumanoidManipulateBlockDiscreteAsynchronous-v0")
    env.reset()

    force_penalties = []
    tendon_penalties = []
    for i in tqdm.tqdm(range(100000)):
        o, r, d, _ = env.step(env.action_space.sample())

        tendon_penalties.append(calculate_tendon_stress_penalty(env.data))
        force_penalties.append(calculate_force_penalty(env.data))

        if d:
            env.reset()

    print(f"Mean Force Penalty: {np.mean(force_penalties)}")
    print(f"Mean Tendon Penalty: {np.mean(tendon_penalties)}")
    print(f"Suggested Scaling: {np.round(np.mean(force_penalties) / np.mean(tendon_penalties))}")
