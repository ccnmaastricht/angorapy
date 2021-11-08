import gym
from gym.spaces import MultiDiscrete
import environments

from tqdm import tqdm

envs = []
for env_spec in tqdm(gym.envs.registry.all()):
    try:
        env = gym.make(env_spec.id)
        info_tuple = (env_spec.id, env.observation_space, env.action_space)
        envs.append(info_tuple)
        if isinstance(info_tuple[2], MultiDiscrete):
            print(info_tuple)
    except:
        pass
