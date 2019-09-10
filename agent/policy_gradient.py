#!/usr/bin/env python
"""Policy Gradient Algorithms.

Currently including REINFORCE and Actor Critic REINFORCE.
"""
import statistics

import numpy

import tensorflow as tf

from agent.core import _RLAgent


class REINFORCEAgent(_RLAgent):

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        # ENVIRONMENT
        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        # TRAINING PARAMETERS
        self.discount = 0.99
        self.learning_rate = 0.01

        # MODEL
        self.actor = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_dimensionality, activation="relu"))
        model.add(tf.keras.layers.Dense(24, activation="relu"))
        model.add(tf.keras.layers.Dense(self.n_actions))
        model.add(tf.keras.layers.Softmax())

        return model

    def loss(self, action_probability):
        return -tf.math.log(action_probability)

    def act(self, state):
        probabilities = self.actor(state)
        action = numpy.random.choice(list(range(self.n_actions)), p=probabilities[0])

        return action, probabilities[0][action]

    def drill(self, env, n_episodes):
        episode_reward_history = []
        for episode in range(n_episodes):
            partial_episode_gradients = []
            reward_trajectory = []

            state = numpy.reshape(env.reset(), [1, -1])
            done = False
            while not done:
                # choose action and calculate partial loss (not yet weighted on future reward)
                with tf.GradientTape() as tape:
                    action, action_probability = self.act(state)
                    partial_loss = self.loss(action_probability)

                # get and remember unweighted gradient
                partial_episode_gradients.append(tape.gradient(partial_loss, self.actor.trainable_variables))

                # actually apply the chosen action
                observation, reward, done, _ = env.step(action)
                observation = numpy.reshape(observation, [1, -1])
                reward_trajectory.append(reward)

                if not done:
                    # next state is observation after executing the action
                    state = observation

            # gather future rewards and apply them to partial gradients
            discounted_future_rewards = [self.discount ** t * sum(reward_trajectory[t:]) for t in
                                         range(len(reward_trajectory))]
            full_gradients = [
                [gradient_tensor * discounted_future_rewards[t]
                 for gradient_tensor in partial_episode_gradients[t]] for t in range(len(discounted_future_rewards))
            ]

            # sum gradients over time steps and optimize the policy based on all time steps
            accumulated_gradients = [
                tf.add_n([full_gradients[t][i_grad]
                          for t in range(len(full_gradients))]) for i_grad in range(len(full_gradients[0]))]
            self.optimizer.apply_gradients(zip(accumulated_gradients, self.actor.trainable_variables))

            # report performance
            episode_reward_history.append(sum(reward_trajectory))
            if episode % 30 == 0 and len(episode_reward_history) > 0:
                print(f"Episode {episode:10d}/{n_episodes}"
                      f" | Mean over last 30: {statistics.mean(episode_reward_history[-30:]):4.2f}")


class ActorCriticREINFORCEAgent(REINFORCEAgent):

    def __init__(self, state_dimensionality, n_actions):
        super().__init__(state_dimensionality, n_actions)

        self.critic_lr = 0.001
        self.critic = self._build_value_model()
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=self.critic_lr)

    def _build_value_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_dimensionality, activation="relu"))
        model.add(tf.keras.layers.Dense(24, activation="relu"))
        model.add(tf.keras.layers.Dense(1))

        return model

    def critic_loss(self, value_prediction, future_reward):
        return future_reward - value_prediction

    def judge_state_value(self, state):
        return self.critic(state)

    def drill(self, env, n_episodes):
        episode_reward_history = []
        for episode in range(n_episodes):
            partial_episode_gradients = []
            state_trajectory = []
            reward_trajectory = []

            state = numpy.reshape(env.reset(), [1, -1])
            done = False
            while not done:
                # choose action and calculate partial loss (not yet weighted on future reward)
                with tf.GradientTape() as tape:
                    action, action_probability = self.act(state)
                    partial_loss = self.loss(action_probability)

                # get and remember unweighted gradient
                partial_episode_gradients.append(tape.gradient(partial_loss, self.actor.trainable_variables))

                # remember the states for later value prediction
                state_trajectory.append(state.copy())

                # actually apply the chosen action
                observation, reward, done, _ = env.step(action)
                observation = numpy.reshape(observation, [1, -1])
                reward_trajectory.append(reward)

                if not done:
                    state = observation

            # make state value predictions and immediately calculate loss and gradients of critic network
            critic_gradients = []
            state_value_predictions = []
            for t, state in enumerate(state_trajectory):
                with tf.GradientTape() as tape:
                    state_value_prediction = self.judge_state_value(state)
                    loss = self.critic_loss(state_value_prediction, sum(reward_trajectory[t:]))
                    state_value_predictions.append(state_value_prediction)

                critic_gradients.append(tape.gradient(loss, self.critic.trainable_variables))

            # gather future rewards and calculate advantages
            advantages = [self.discount ** t * (sum(reward_trajectory[t:]) - state_value_predictions[t][0]) for t in
                          range(len(reward_trajectory))]
            full_gradients = [[gradient_tensor * advantages[t]
                               for gradient_tensor in partial_episode_gradients[t]] for t in range(len(advantages))]

            # sum gradients over time steps and optimize the policy based on all time steps
            accumulated_gradients = [
                tf.add_n([full_gradients[t][i_grad]
                          for t in range(len(full_gradients))]) for i_grad in range(len(full_gradients[0]))]
            self.optimizer.apply_gradients(zip(accumulated_gradients, self.actor.trainable_variables))

            # sum critic's gradients over time steps and optimize it based on all time steps
            accumulated_critic_gradients = [
                tf.add_n([critic_gradients[t][i_grad]
                          for t in range(len(critic_gradients))]) for i_grad in range(len(critic_gradients[0]))]
            self.critic_optimizer.apply_gradients(zip(accumulated_critic_gradients, self.critic.trainable_variables))

            # report performance
            episode_reward_history.append(sum(reward_trajectory))
            if episode % 30 == 0 and len(episode_reward_history) > 0:
                print(
                    f"Episode {episode:10d}/{n_episodes} | Mean over last 30: {statistics.mean(episode_reward_history[-30:]):4.2f}")

class TRPOAgent:

    def __init__(self):
        super().__init__()
        raise NotImplementedError


class PPOAgent:

    def __init__(self):
        super().__init__()

        self.actor = self._build_model()
        self.critic = self._build_critic_model()

    def _build_model(self):
        pass

    def _build_critic_model(self):
        pass
