#!/usr/bin/env python
"""TODO Module Docstring."""
import time

import numpy
import tensorflow as tf

from agent.core import get_discounted_returns


class REINFORCEAgent:

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        # ENVIRONMENT
        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        # TRAINING PARAMETERS
        self.learning_rate = 0.001
        self.discount_factor = tf.constant(0.999, dtype=tf.float64)

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

    @staticmethod
    def _loss(action_probability):
        return -tf.math.log(action_probability)

    def act(self, state):
        probabilities = self.actor(state)
        action = tf.random.categorical(tf.math.log(probabilities), 1)[0][0]

        return action, probabilities[0][action]

    def drill(self, env, n_episodes):
        t_start = time.time()

        episode_reward_history = []
        for episode in range(n_episodes):
            partial_episode_gradients = []
            reward_trajectory = []

            state = tf.reshape(env.reset(), [1, -1])
            done = False
            while not done:
                # choose action and calculate partial loss (not yet weighted on future reward)
                with tf.GradientTape() as tape:
                    action, action_probability = self.act(state)
                    partial_loss = self._loss(action_probability)

                # get and remember unweighted gradient
                partial_episode_gradients.append(tape.gradient(partial_loss, self.actor.trainable_variables))

                # actually apply the chosen action
                observation, reward, done, _ = env.step(action.numpy())
                observation = tf.reshape(observation, [1, -1])

                reward_trajectory.append(reward)

                if not done:
                    # next state is observation after executing the action
                    state = observation

            # gather future rewards and apply them to partial gradients
            discounted_returns = get_discounted_returns(reward_trajectory, self.discount_factor)
            full_gradients = [[tf.math.scalar_mul(discounted_returns[t], gradient_tensor)
                               for gradient_tensor in partial_episode_gradients[t]] for t in
                              range(len(discounted_returns))]

            # sum gradients over time steps and optimize the policy based on all time steps
            accumulated_gradients = [
                tf.math.add_n([full_gradients[t][i_grad]
                               for t in range(len(full_gradients))]) for i_grad in range(len(full_gradients[0]))]
            self.optimizer.apply_gradients(zip(accumulated_gradients, self.actor.trainable_variables))

            # report performance
            episode_reward_history.append(sum(reward_trajectory))
            if episode % 30 == 0 and len(episode_reward_history) > 0:
                print(f"Episode {episode:10d}/{n_episodes}"
                      f" | Mean over last 30: {statistics.mean(episode_reward_history[-30:]):4.2f}"
                      f" | ExecTime: {round(time.time() - t_start, 2)}")
                t_start = time.time()


class ActorCriticREINFORCEAgent(REINFORCEAgent):

    def __init__(self, state_dimensionality, n_actions):
        super().__init__(state_dimensionality, n_actions)

        self.critic_lr = 0.001
        self.critic = self._build_value_model()
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=self.critic_lr)

        self.discount_factor = 0.999

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
                    partial_loss = self._loss(action_probability)

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
                    state_value_predictions.append(state_value_prediction[0][0].numpy())

                critic_gradients.append(tape.gradient(loss, self.critic.trainable_variables))

            # gather future rewards and calculate advantages
            discounted_returns = get_discounted_returns(reward_trajectory, self.discount_factor)
            advantages = numpy.subtract(discounted_returns, state_value_predictions)

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