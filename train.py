import logging
import os

from gym.spaces import Box

from agent.ppo import PPOAgent
from environments import *
from models.fully_connected import build_ffn_distinct_models
from utilities.const import COLORS
from utilities.story import StoryTeller
from utilities.util import env_extract_dims, set_all_seeds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SETTINGS
DEBUG = False
GPU = True
EXPORT_TO_FILE = False  # if true, saves/reads policy to be loaded in workers into file
LOAD_ID = None
SEPERATE_EVAL = True if not DEBUG else False

TASK = "LunarLanderContinuous-v2"
build_models = build_ffn_distinct_models

ITERATIONS = 1000
WORKERS = 8
HORIZON = 2048 if not DEBUG else 128
EPOCHS = 5 if not DEBUG else 1
BATCH_SIZE = 512

LEARNING_RATE_POLICY = 0.0003
LEARNING_RATE_CRITIC = 0.001
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.97
EPSILON_CLIP = 0.2
C_ENTROPY = 0.01
C_VALUE = 1
GRADIENT_CLIP_NORM = 0.5
CLIP_VALUE = True

if DEBUG:
    logging.warning("YOU ARE RUNNING IN DEBUG MODE!")

# setup environment and extract and report information
env = gym.make(TASK)
state_dimensionality, number_of_actions = env_extract_dims(env)
env_action_space_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
env_observation_space_type = "continuous" if isinstance(env.observation_space, Box) else "discrete"
env_name = env.unwrapped.spec.id

# announce experiment
bc, ec, wn = COLORS["HEADER"], COLORS["ENDC"], COLORS["WARNING"]
print(f"{wn}-----------------------------------------{ec}\n"
      f"Learning the Task: {bc}{env_name}{ec}\n"
      f"{bc}{state_dimensionality}{ec}-dimensional states ({bc}{env_observation_space_type}{ec}) "
      f"and {bc}{number_of_actions}{ec} actions ({bc}{env_action_space_type}{ec}).\n"
      f"{wn}-----------------------------------------{ec}\n")


if not GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if LOAD_ID is not None:
    print(f"Loading from state {LOAD_ID}")
    agent = PPOAgent.from_agent_state(LOAD_ID)
else:
    # set up the agent and a reporting module
    agent = PPOAgent(build_models, env,
                     horizon=HORIZON,
                     workers=WORKERS,
                     learning_rate_pi=LEARNING_RATE_POLICY,
                     learning_rate_v=LEARNING_RATE_CRITIC,
                     discount=DISCOUNT_FACTOR,
                     clip=EPSILON_CLIP,
                     c_entropy=C_ENTROPY,
                     c_value=C_VALUE,
                     lam=GAE_LAMBDA,
                     gradient_clipping=GRADIENT_CLIP_NORM,
                     clip_values=CLIP_VALUE)

    print(f"Created agent with ID {agent.agent_id}")

agent.set_gpu(GPU)
teller = StoryTeller(agent, env, frequency=0)

# train
agent.drill(n=ITERATIONS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            story_teller=teller,
            export=EXPORT_TO_FILE,
            save_every=0,
            separate_eval=SEPERATE_EVAL)

agent.save_agent_state()
agent.evaluate(n=1, render=True)

env.close()
