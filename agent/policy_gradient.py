#!/usr/bin/env python
"""Policy Gradient Algorithms.

Currently including REINFORCE and Actor Critic REINFORCE.
"""
import statistics
from typing import Tuple, List

import numpy
import tensorflow as tf

from agent.core import _RLAgent
from policy_networks.fully_connected import PPOActorCriticNetwork


def get_discounted_returns(reward_trajectory, discount_factor):
    return [sum([discount_factor ** k * r for k, r in enumerate(reward_trajectory[t:])]) for t in
            range(len(reward_trajectory))]


class REINFORCEAgent:

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        # ENVIRONMENT
        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        # TRAINING PARAMETERS
        self.learning_rate = 0.001

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
                    partial_loss = self._loss(action_probability)

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
            discounted_returns = self._get_discounted_returns(reward_trajectory)
            full_gradients = [[gradient_tensor * discounted_returns[t]
                               for gradient_tensor in partial_episode_gradients[t]] for t in
                              range(len(discounted_returns))]

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
            discounted_returns = self._get_discounted_returns(reward_trajectory)
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


class PPOAgent(_RLAgent):

    def __init__(self, state_dimensionality, n_actions):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        # learning parameters
        self.discount = 0.99
        self.learning_rate = 0.001
        self.epsilon_clip = 0.2

        # Models
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        model = PPOActorCriticNetwork(self.state_dimensionality, self.n_actions)
        return model

    def act(self, state) -> Tuple[int, float]:
        probabilities, _ = self.model(state)
        action = numpy.random.choice(list(range(self.n_actions)), p=probabilities[0])

        return action, probabilities[0][action]

    def gather_experience(self, env, n_trajectories: int) -> Tuple[List, List, List, List]:
        state_trajectories = []
        reward_trajectories = []
        action_trajectories = []
        action_probability_trajectories = []

        for episode in range(n_trajectories):
            state_trajectory = []
            reward_trajectory = []
            action_trajectory = []
            action_probability_trajectory = []

            done = False
            state = tf.reshape(env.reset(), [1, -1])
            while not done:
                state_trajectory.append(state)  # does not incorporate the state inducing DONE

                action, action_probability = self.act(state)
                observation, reward, done, _ = env.step(action)
                reward_trajectory.append(reward)
                action_trajectory.append(action)
                action_probability_trajectory.append(action_probability)

                state = tf.reshape(observation, [1, -1])

            state_trajectories.append(state_trajectory)
            reward_trajectories.append(reward_trajectory)
            action_probability_trajectories.append(action_probability_trajectory)
            action_trajectories.append(action_trajectory)

        return state_trajectories, reward_trajectories, action_trajectories, action_probability_trajectories

    def train_actor(self, states, advantages):
        """Train the actor network.

        Optimization is performed given a batch of state trajectories and the corresponding batch of advantage
        estimations.
        """
        pass

    def train_critic(self, states, advantages):
        """Train the critic network.

        Optimization is performed given a batch of state trajectories and the corresponding batch of advantage
        estimations.
        """
        pass

    def clipping_loss(self, old_action_prob, action_prob, advantage):
        r = (action_prob / old_action_prob)
        return tf.minimum(
            r * advantage,
            tf.clip_by_value(r, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage
        )

    @staticmethod
    def critic_loss(prediction, target):
        return tf.square(prediction - target)

    def entropy_bonus(self):
        pass

    def drill(self, env, epochs, N):
        for epoch in range(epochs):
            # run simulations
            state_trajectories, reward_trajectories, action_trajectories, action_prob_trajectories \
                = self.gather_experience(env, N)

            print(f"Epoch {epoch}: Average reward of {statistics.mean([sum(rt) for rt in reward_trajectories])}")

            discounted_returns = [get_discounted_returns(reward_trajectory, self.discount) for reward_trajectory in
                                  reward_trajectories]
            state_value_predictions = [[self.model.predict(state)[1][0][0] for state in trajectory] for trajectory in
                                       state_trajectories]
            advantages = [tf.dtypes.cast(tf.subtract(disco_traj, value_traj), tf.float64) for disco_traj, value_traj in
                          zip(discounted_returns, state_value_predictions)]

            for trajectory_id in range(len(state_trajectories)):
                for t in range(len(state_trajectories[trajectory_id])):
                    with tf.GradientTape() as tape:
                        state = state_trajectories[trajectory_id][t]
                        discounted_return = discounted_returns[trajectory_id][t]
                        advantage = advantages[trajectory_id][t]
                        action = action_trajectories[trajectory_id][t]
                        old_action_prob = action_prob_trajectories[trajectory_id][t]

                        action_probabilities, state_value = self.model(state)

                        total_loss = self.critic_loss(state_value, discounted_return) \
                                     - self.clipping_loss(old_action_prob,
                                                          action_probabilities[0][action],
                                                          advantage)

                    gradients = tape.gradient(total_loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))