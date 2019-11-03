import os

from gym.spaces import Box

from agent.ppo import PPOAgent
from environments import *
from models.fully_connected import build_ffn_shared_models, build_ffn_distinct_models
from utilities.util import env_extract_dims, set_all_seeds
from utilities.story import StoryTeller

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

set_all_seeds(1)

# SETTINGS
DEBUG = False
GPU = False
EXPORT_TO_FILE = False  # if true, saves/reads policy to be loaded in workers into file
LOAD_ID = None

TASK = "LunarLander-v2"
build_models = build_ffn_distinct_models

ITERATIONS = 1000
WORKERS = 1
HORIZON = 2048 if not DEBUG else 128
EPOCHS = 4
BATCH_SIZE = 512

LEARNING_RATE_POLICY = 3e-4
LEARNING_RATE_CRITIC = 1e-3
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95
EPSILON_CLIP = 0.2
C_ENTROPY = 0.0
C_VALUE = 0.5
GRADIENT_CLIP_NORM = 0.5
CLIP_VALUE = True

# setup environment and extract and report information
env = gym.make(TASK)
state_dimensionality, number_of_actions = env_extract_dims(env)
env_action_space_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
env_observation_space_type = "continuous" if isinstance(env.observation_space, Box) else "discrete"
env_name = env.unwrapped.spec.id
print(f"-----------------------------------------\n"
      f"Learning the Task: {env_name}\n"
      f"{state_dimensionality}-dimensional states ({env_observation_space_type}) "
      f"and {number_of_actions} actions ({env_action_space_type}).\n"
      f"-----------------------------------------\n")
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
            export_to_file=EXPORT_TO_FILE,
            save_every=0)

agent.save_agent_state()

env.close()
