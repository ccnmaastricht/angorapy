#!/usr/bin/env python
"""Policy Gradient Algorithms.

Currently including REINFORCE and Actor Critic REINFORCE.
"""
import itertools
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
        self.model = PPOActorCriticNetwork(self.state_dimensionality, self.n_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def act(self, state) -> Tuple[int, float]:
        probabilities, _ = self.model(state)
        action = numpy.random.choice(list(range(self.n_actions)), p=probabilities[0])

        return action, probabilities[0][action]

    def gather_experience(self, env, n_trajectories: int) -> Tuple[List, List, List, List]:
        """Gather experience in an environment for n trajectories.

        :param env:                 the environment in which the trajectories will be produced
        :param n_trajectories:      the number of desired trajectories

        :return:                    a 4-tuple where each element is a list of trajectories of s, r, a and p(a)
        """
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
                action, action_probability = self.act(state)
                observation, reward, done, _ = env.step(action)

                # remember experience
                state_trajectory.append(tf.reshape(state, [-1]))  # does not incorporate the state inducing DONE
                reward_trajectory.append(reward)
                action_trajectory.append(action)
                action_probability_trajectory.append(action_probability)

                # next state
                state = tf.reshape(observation, [1, -1])

            state_trajectories.append(state_trajectory)
            reward_trajectories.append(reward_trajectory)
            action_probability_trajectories.append(action_probability_trajectory)
            action_trajectories.append(action_trajectory)

        return state_trajectories, reward_trajectories, action_trajectories, action_probability_trajectories

    def _actor_objective(self, old_action_prob, action_prob, advantage):
        r = (action_prob / old_action_prob)
        return tf.minimum(
            r * advantage,
            tf.clip_by_value(r, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage
        )

    @staticmethod
    def _critic_loss(prediction, target):
        return tf.square(prediction - target)

    def entropy_bonus(self):
        """TODO add entropy bonus for objective"""
        pass

    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> None:
        """Optimize the agents policy/value network based on a given dataset.

        :param dataset:         tensorflow dataset containing s, a, p(a), r and A as components per data point
        :param epochs:          number of epochs to train on this dataset
        :param batch_size:      batch size with which the dataset is sampled
        """
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break bias, then divided into batches
            shuffled_dataset = dataset.shuffle(1000)  # TODO appropriate buffer size based on number of datapoints
            batched_dataset = shuffled_dataset.batch(batch_size)

            for batch in batched_dataset:
                with tf.GradientTape() as tape:
                    action_probabilities, state_value = self.model(batch["state"], training=True)

                    # loss needs to be negated since the original objective from the PPO paper is for maximization
                    loss = - (self._actor_objective(batch["action_prob"],
                                                    [action_probabilities[i][a] for i, a in
                                                     enumerate(batch["action"])],
                                                    batch["advantage"]) - self._critic_loss(state_value,
                                                                                            batch["return"]))

                # calculate and apply gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def drill(self, env, iterations: int, epochs: int, agents: int, batch_size: int):
        """Main training loop of the agent.

        Runs **iterations** cycles of experience gathering and optimization based on the gathered experience.

        :param env:             the environment on which the agent should be drilled
        :param iterations:      the number of experience-optimization cycles that shall be run
        :param epochs:          the number of epochs for which the model is optimized on the same experience data
        :param agents:          the number of trajectories generated during the experience gathering
        :param batch_size       batch size for the optimization

        :return:                self
        """
        for iteration in range(iterations):
            # run simulations
            s_trajectories, r_trajectories, a_trajectories, a_prob_trajectories = self.gather_experience(env, agents)

            print(f"Iteration {iteration}: Average reward of {statistics.mean([sum(r) for r in r_trajectories])}")

            discounted_returns = [tf.dtypes.cast(get_discounted_returns(reward_trajectory, self.discount), tf.float64)
                                  for reward_trajectory in
                                  r_trajectories]
            state_value_predictions = [[self.model.predict(tf.reshape(state, [1, -1]))[1][0][0] for state in trajectory]
                                       for trajectory in
                                       s_trajectories]
            advantages = [tf.dtypes.cast(tf.subtract(disco_traj, value_traj), tf.float64) for disco_traj, value_traj in
                          zip(discounted_returns, state_value_predictions)]

            # make tensorflow data set for faster data access during training
            dataset = tf.data.Dataset.from_tensor_slices({
                "state": list(itertools.chain(*s_trajectories)),
                "action": list(itertools.chain(*a_trajectories)),
                "action_prob": list(itertools.chain(*a_prob_trajectories)),
                "return": list(itertools.chain(*discounted_returns)),
                "advantage": list(itertools.chain(*advantages))
            })

            # use the dataset to optimize the model
            self.optimize_model(dataset, epochs, batch_size)

        return self
