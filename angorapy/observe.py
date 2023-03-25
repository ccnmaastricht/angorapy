#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from angorapy.analysis.investigators import Investigator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from angorapy.environments.hand.shadowhand import FINGERTIP_SITE_NAMES, BaseShadowHandEnv

import time

from agent.ppo_agent import PPOAgent
from common.const import BASE_SAVE_PATH, PATH_TO_EXPERIMENTS
from common.wrappers import make_env
import tensorflow as tf

tf.get_logger().setLevel('INFO')

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=None)
parser.add_argument("--env", type=str, nargs="?", help="force testing environment", default="")
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")
parser.add_argument("--force-case-circulation", action="store_true", help="circle through goal definitions")
parser.add_argument("--freeze-wrist", action="store_true", help="prevent wrist movements")
parser.add_argument("--hide-targets", action="store_true", help="do not show visualization of targets")
parser.add_argument("--rcon", type=str, help="reward configuration", default=None)
parser.add_argument("--act-stochastic", action="store_true", help="keep stochasticity in decisions")

args = parser.parse_args()

scale_the_substeps = False

if args.state not in ["b", "best", "last"]:
    args.state = int(args.state)

if args.id is None:
    ids = map(int, [d for d in os.listdir(BASE_SAVE_PATH) if os.path.isdir(os.path.join(BASE_SAVE_PATH, d))])
    args.id = max(ids)

start = time.time()
agent = PPOAgent.from_agent_state(args.id, args.state, force_env_name=None if not args.env else args.env)
print(f"Agent {args.id} successfully loaded.")

try:
    tf.keras.utils.plot_model(agent.joint, to_file=f"{PATH_TO_EXPERIMENTS}/{args.id}/model.png", expand_nested=True,
                              show_shapes=True, dpi=300)
except:
    print("Could not create model plot.")

investigator = Investigator.from_agent(agent)
env = make_env(agent.env.spec.id, transformers=agent.env.transformers, render_mode="human")
if args.freeze_wrist:
    env.toggle_wrist_freezing()

substeps = "" if not hasattr(env.unwrapped, "sim") else f" with {env.unwrapped.sim.nsubsteps} substeps"
print(f"Evaluating on {env.unwrapped.spec.id}{substeps}.")
print(f"Environment has the following transformers: {env.transformers}")

# if args.env != "":
#     env = make_env(args.env, args.rcon)
# elif scale_the_substeps:
#     parts = env.env.unwrapped.spec.id.split("-")
#     new_name = parts[0] + "Fine" + "-" + parts[1]
#     print(new_name)
#     env = make_env(new_name, args.rcon)

if isinstance(env.unwrapped, BaseShadowHandEnv):
    env.set_delta_t_simulation(0.002)
    env.set_original_n_substeps_to_sspcs()
    env.change_color_scheme("default")
    env.change_perspective("topdown-far")
if not args.force_case_circulation or ("Reach" not in env.unwrapped.spec.id):
    for i in range(1000):
        investigator.render_episode(env, act_confidently=not args.act_stochastic)
# else:
#     env = make_env("ReachFFAbsolute-v0", transformers=agent.env.transformers)
#     if args.freeze_wrist:
#         env.env.toggle_wrist_freezing()
#
#     for i in range(1000):
#         print(f"\nNew Episode, finger {FINGERTIP_SITE_NAMES[i % 4]}")
#         env.forced_finger = i % 4
#         env.env.forced_finger = i % 4
#         env.unwrapped.forced_finger = i % 4
#         investigator.render_episode(env, slow_down=False)