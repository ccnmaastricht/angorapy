import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('InvertedPendulum-v2')

k = np.array([[-3.16,  -12.89, -113.58,  -43.19]]) # <- negative definiteness based on Jacobian matrix


new_state = env.reset()
count = 0
done = False
r = np.array([[2, 0, 0, 0]]).T       # reference position is taken 2
angle = []
position = []
actions = []
rewards = []
while not done:
    angle.append(new_state[2])
    position.append(new_state[0])
    error = r - new_state.reshape((4, 1))
    action = np.dot(k, error)
    actions.append(action[0])
    new_state, reward, done, info = env.step(min(-10, max(10, action[0])))
    # env.render()
    rewards.append(reward)
    count+=1
    if count>600:
        break
env.close()

plt.plot(angle)
plt.xlabel('Timesteps')
plt.ylabel('Pole Angle')
plt.show()