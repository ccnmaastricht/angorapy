#!/usr/bin/env python
"""TODO Module Docstring."""
import collections
import itertools
import random
import statistics
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy
import tensorflow as tf
from tensorflow import keras

from agent.core import RLAgent, get_discounted_returns
from agent.gather import Gatherer
from agent.ppo import PPOAgent
from utilities.datatypes import Experience


@DeprecationWarning
class PPOAgentJoint(PPOAgent):
    """Agent using the Proximal Policy Optimization Algorithm for learning."""

    def __init__(self, policy: tf.keras.Model, gatherer, learning_rate: float, discount: float,
                 epsilon_clip: float):
        """Initialize the Agent.

        :param learning_rate:           the agents learning rate
        :param discount:                discount factor applied to future rewards
        :param epsilon_clip:            clipping range for the actor's objective
        """
        super().__init__(gatherer, learning_rate, discount, epsilon_clip)

        # learning parameters
        self.discount = tf.constant(discount, dtype=tf.float64)
        self.learning_rate = learning_rate
        self.epsilon_clip = tf.constant(epsilon_clip, dtype=tf.float64)
        self.c_entropy = tf.constant(0, dtype=tf.float64)

        # Models
        self.policy = policy
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def full_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.policy(state, training=training)

    def critic_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.policy(state, training=training)[1]

    def actor_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.policy(state, training=training)[0]

    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> List[float]:
        """Optimize the agents policy/value network based on a given dataset.

        :param dataset:         tensorflow dataset containing s, a, p(a), r and A as components per data point
        :param epochs:          number of epochs to train on this dataset
        :param batch_size:      batch size with which the dataset is sampled
        """
        loss_history = []
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break bias, then divided into batches
            shuffled_dataset = dataset.shuffle(10000)  # TODO appropriate buffer size based on number of datapoints
            batched_dataset = shuffled_dataset.batch(batch_size)

            for batch in batched_dataset:
                # use the dataset to optimize the model
                with tf.device(self.device):
                    with tf.GradientTape() as tape:
                        action_probabilities, state_value = self.full_prediction(batch["state"], training=True)

                        loss = self.joint_loss_function(batch["action_prob"],
                                                        tf.convert_to_tensor([action_probabilities[i][a] for i, a
                                                                              in enumerate(batch["action"])],
                                                                             dtype=tf.float64),
                                                        batch["advantage"],
                                                        state_value,
                                                        batch["return"],
                                                        action_probs=action_probabilities)

                    loss_history.append(loss.numpy().item())

                    # calculate and apply gradients
                    gradients = tape.gradient(loss, self.policy.trainable_variables)
                    self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return loss_history


@DeprecationWarning
class EpisodicGatherer(Gatherer):

    def gather(self, agent) -> Tuple[List, List, List, List, List]:
        """Gather experience in an environment for n trajectories.

        :param agent:               the agent who is to be set into the environment

        :return:                    a 4-tuple where each element is a list of trajectories of s, r, a and p(a)
        """
        self.last_episodes_completed = 0

        state_trajectories = []
        reward_trajectories = []
        action_trajectories = []
        action_probability_trajectories = []

        for episode in range(self.n_trajectories):
            state_trajectory = []
            reward_trajectory = []
            action_trajectory = []
            action_probability_trajectory = []

            done = False
            state = tf.reshape(self.env.reset(), [1, -1])
            while not done:
                action, action_probability = agent.act(state)
                observation, reward, done, _ = self.env.step(action.numpy())

                # remember experience
                state_trajectory.append(tf.reshape(state, [-1]))  # does not incorporate the state inducing DONE
                reward_trajectory.append(reward)
                action_trajectory.append(action)
                action_probability_trajectory.append(action_probability)

                # next state
                state = tf.reshape(observation, [1, -1])

            self.last_episodes_completed += 1

            state_trajectories.append(state_trajectory)
            reward_trajectories.append(reward_trajectory)
            action_probability_trajectories.append(action_probability_trajectory)
            action_trajectories.append(action_trajectory)

        state_value_predictions = [[agent.critic_prediction(tf.reshape(state, [1, -1]))[0][0]
                                    for state in trajectory] for trajectory in state_trajectories]
        advantages = [generalized_advantage_estimator(reward_trajectory, value_predictions,
                                                      gamma=agent.discount, gae_lambda=0.95)
                      for reward_trajectory, value_predictions in zip(reward_trajectories, state_value_predictions)]
        returns = numpy.add(advantages, state_value_predictions)

        # create the tensorflow dataset
        return tf.data.Dataset.from_tensor_slices({
            "state": list(itertools.chain(*state_trajectories)),
            "action": list(itertools.chain(*action_trajectories)),
            "action_prob": list(itertools.chain(*action_probability_trajectories)),
            "return": list(itertools.chain(*returns)),
            "advantage": list(itertools.chain(*advantages))
        })


@DeprecationWarning
class _QLearningAgent(RLAgent, ABC):
    """Base Class for Q Learning Agents.
    Contains most functionality s.t. implementations only need to add model building.
    """

    def __init__(self, state_dimensionality, n_actions, eps_init=0.1, eps_decay=0.999, eps_min=0.001):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        self.learning_rate = 0.001
        self.gamma = 0.95

        # epsilon greedy exploration parameters
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.epsilon = eps_init

        # function approximation model
        self.model = self._build_model()

        # replay buffer
        self.memory = collections.deque(maxlen=2000)

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def remember(self, experience, done):
        self.memory.append((experience, done))

    def act(self, state: numpy.ndarray):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            q_values = self.model.predict(state)
            return numpy.argmax(q_values)

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for experience, done in minibatch:
            if not done:
                # target for action taken by agent is predicted value of next state (when following policy) + reward
                focused_target = experience.reward + self.gamma * numpy.amax(
                    self.model.predict(experience.observation)[0])
            else:
                focused_target = experience.reward

            # target is current models prediction for all actions not taken by the agent -> no influence on loss
            target_vector = self.model.predict(experience.state)
            target_vector[0][experience.action] = focused_target

            # store in batch
            states.append(experience.state.reshape(-1))
            targets.append(target_vector.reshape(-1))

        # learn on batch
        self.model.fit(numpy.array(states), numpy.array(targets), epochs=1, verbose=0)

    def drill(self, env, n_iterations, batch_size=32, print_every=1000, avg_over=20):
        episode_rewards = [0]
        state = numpy.reshape(env.reset(), [1, -1])
        for iteration in range(n_iterations):
            # choose an action and make a step in the environment, based on the agents current knowledge
            action = self.act(numpy.array(state))
            observation, reward, done, info = env.step(action)
            episode_rewards[-1] += reward

            # reshape observation for tensorflow
            observation = numpy.reshape(observation, [1, -1])

            # remember experience made during the step in the agents memory
            self.remember(Experience(state, action, reward, observation), done)

            # log performance
            if iteration % print_every == 0:
                print(f"Iteration {iteration:10d}/{n_iterations} [{round(iteration/n_iterations * 100, 0)}%]"
                      f" | {len(episode_rewards) - 1:4d} episodes done"
                      f" | last {min(avg_over, len(episode_rewards)):2d}: "
                      f"avg {0 if len(episode_rewards) <= 1 else statistics.mean(episode_rewards[-avg_over:-1]):5.2f}; "
                      f"max {0 if len(episode_rewards) <= 1 else max(episode_rewards[-avg_over:-1]):5.2f}; "
                      f"min {0 if len(episode_rewards) <= 1 else min(episode_rewards[-avg_over:-1]):5.2f}"
                      f" | epsilon: {self.epsilon}")

            if len(self.memory) > batch_size:
                # learn from memory
                self.learn(batch_size)
                # update exploration
                self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

            # if episode ended reset env and rewards, otherwise go to next state
            if done:
                state = numpy.reshape(env.reset(), [1, -1])
                episode_rewards.append(0)
            else:
                state = observation


@DeprecationWarning
class LinearQLearningAgent(_QLearningAgent):

    def __init__(self, state_dimensionality, n_actions):
        super(LinearQLearningAgent, self).__init__(state_dimensionality, n_actions)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(self.n_actions, input_dim=self.state_dimensionality, activation="linear"))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model


@DeprecationWarning
class DeepQLearningAgent(_QLearningAgent):

    def __init__(self, state_dimensionality, n_actions):
        super(DeepQLearningAgent, self).__init__(state_dimensionality, n_actions)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_dimensionality, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.n_actions, activation="linear"))

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model


@DeprecationWarning
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


@DeprecationWarning
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