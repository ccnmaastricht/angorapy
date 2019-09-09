import os

import statistics

import gym
import numpy

from agent.policy_gradient import ActorCriticREINFORCEAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# activate eager execution to get rid of bullshit static graphs
tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx("float64")  # prevent precision issues

# SETTINGS
TOTAL_EPISODES = 100000

# ENVIRONMENT
env = gym.make("CartPole-v0")
number_of_actions = env.action_space.n
state_dimensionality = env.observation_space.shape[0]

# AGENT
agent = ActorCriticREINFORCEAgent(state_dimensionality, number_of_actions)

# TRAINING
episode_reward_history = []
for episode in range(TOTAL_EPISODES):
    partial_episode_gradients = []
    state_trajectory = []
    reward_trajectory = []

    state = numpy.reshape(env.reset(), [1, -1])
    done = False
    while not done:
        # choose action and calculate partial loss (not yet weighted on future reward)
        with tf.GradientTape() as tape:
            action, action_probability = agent.act(state)
            partial_loss = agent.loss(action_probability)

        # get and remember unweighted gradient
        partial_episode_gradients.append(tape.gradient(partial_loss, agent.model.trainable_variables))

        # remember the states for later value prediction
        state_trajectory.append(state.copy())

        # actually apply the chosen action
        observation, reward, done, _ = env.step(action)
        observation = numpy.reshape(observation, [1, -1])
        reward_trajectory.append(reward)

        if not done:
            # next state is observation after executing the action
            state = observation

    # make state value predictions and immediately calculate loss and gradients of critic network
    critic_gradients = []
    state_value_predictions = []
    for t, state in enumerate(state_trajectory):
        with tf.GradientTape() as tape:
            state_value_prediction = agent.judge_state_value(state)
            loss = agent.critic_loss(state_value_prediction, sum(reward_trajectory[t:]))
            state_value_predictions.append(state_value_prediction)

        critic_gradients.append(tape.gradient(loss, agent.critic.trainable_variables))

    # gather future rewards and calculate advantages
    advantages = [agent.discount ** t * (sum(reward_trajectory[t:]) - state_value_predictions[t][0]) for t in
                  range(len(reward_trajectory))]
    full_gradients = [
        [gradient_tensor * advantages[t]
         for gradient_tensor in partial_episode_gradients[t]] for t in range(len(advantages))
    ]

    # sum gradients over time steps
    accumulated_gradients = [
        tf.add_n([full_gradients[t][i_grad]
                for t in range(len(full_gradients))]) for i_grad in range(len(full_gradients[0]))
    ]

    # optimize the policy based on all time steps
    agent.optimizer.apply_gradients(zip(accumulated_gradients, agent.model.trainable_variables))

    # sum critic's gradients over time steps
    accumulated_critic_gradients = [
        tf.add_n([critic_gradients[t][i_grad]
                  for t in range(len(critic_gradients))]) for i_grad in range(len(critic_gradients[0]))
    ]

    # optimize the critic based on all time steps
    agent.critic_optimizer.apply_gradients(zip(accumulated_critic_gradients, agent.critic.trainable_variables))

    # report performance
    episode_reward_history.append(sum(reward_trajectory))
    if episode % 30 == 0 and len(episode_reward_history) > 0:
        print(
            f"Episode {episode:10d}/{TOTAL_EPISODES} | Mean over last 30: {statistics.mean(episode_reward_history[-30:]):4.2f}")

env.close()
