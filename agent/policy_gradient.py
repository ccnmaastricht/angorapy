#!/usr/bin/env python
"""Policy Gradient Algorithms.

Currently including REINFORCE and Actor Critic REINFORCE.
"""
import itertools
import statistics
import time
from typing import Tuple, List

import gym
import numpy
import tensorflow as tf

from agent.core import _RLAgent
from policy_networks.fully_connected import PPOActorCriticNetwork

tf.keras.backend.set_floatx("float64")


def get_discounted_returns(reward_trajectory, discount_factor: tf.Tensor):
    # TODO make native tf
    return [tf.math.reduce_sum([tf.math.pow(discount_factor, k) * r for k, r in enumerate(reward_trajectory[t:])]) for t
            in
            tf.range(len(reward_trajectory))]


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


class PPOAgent(_RLAgent):
    """Agent using the Proximal Policy Optimization Algorithm for learning."""

    def __init__(self, state_dimensionality, n_actions, learning_rate: float, discount: float, epsilon_clip: float):
        """Initialize the Agent.

        :param state_dimensionality:    number of dimensions in the states that the agent has to process
        :param n_actions:               number of actions the agent can choose from
        """
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.n_actions = n_actions

        # learning parameters
        self.discount = tf.constant(discount, dtype=tf.float64)
        self.learning_rate = learning_rate
        self.epsilon_clip = tf.constant(epsilon_clip, dtype=tf.float64)

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

    def _actor_objective(self, old_action_prob: tf.Tensor, action_prob: tf.Tensor, advantage: tf.Tensor):
        r = tf.math.divide(action_prob, old_action_prob)
        return tf.minimum(
            tf.math.multiply(r, advantage),
            tf.math.multiply(tf.clip_by_value(r, 1 - self.epsilon_clip, 1 + self.epsilon_clip), advantage)
        )

    @staticmethod
    def _critic_loss(prediction, target):
        return tf.square(prediction - target)

    def entropy_bonus(self):
        """TODO add entropy bonus for objective"""
        pass

    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> None:
        """Optimize the agents policy/value network based on a given dataset.

        Since data processing is apparently not possible with tensorflow data sets on a GPU, we will only let the GPU
        handle the training, but keep the rest of the data pipeline on the CPU. I am not currently sure if this is the
        best approach, but it might be the good anyways for large data chunks anyways due to GPU memory limits. It also
        should not make much of a difference since gym runs entirely on CPU anyways, hence for every experience
        gathering we need to transfer all Tensors from CPU to GPU, no matter whether the dataset is stored on GPU or
        not. Even more so this applies with running simulations on the cluster.

        :param dataset:         tensorflow dataset containing s, a, p(a), r and A as components per data point
        :param epochs:          number of epochs to train on this dataset
        :param batch_size:      batch size with which the dataset is sampled
        """
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break bias, then divided into batches
            shuffled_dataset = dataset.shuffle(10000)  # TODO appropriate buffer size based on number of datapoints
            batched_dataset = shuffled_dataset.batch(batch_size)

            for batch in batched_dataset:
                # use the dataset to optimize the model
                with tf.device("GPU:0"):
                    with tf.GradientTape() as tape:
                        action_probabilities, state_value = self.model(batch["state"], training=True)

                        # loss needs to be negated since the original objective from the PPO paper is for maximization
                        loss = - (self._actor_objective(batch["action_prob"],
                                                        tf.convert_to_tensor([action_probabilities[i][a] for i, a
                                                                              in enumerate(batch["action"])],
                                                                             dtype=tf.float64),
                                                        batch["advantage"]) - self._critic_loss(state_value,
                                                                                                batch["return"]))

                    # calculate and apply gradients
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def drill(self, env: gym.Env, iterations: int, epochs: int, agents: int, batch_size: int):
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
            if iteration > 10:
                self.evaluate(env, 1, True)

            discounted_returns = [get_discounted_returns(reward_trajectory, self.discount) for reward_trajectory in
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

            self.optimize_model(dataset, epochs, batch_size)

        return self

    def evaluate(self, env: gym.Env, n: int, render: bool=False) -> List[int]:
        rewards = []
        for episode in range(n):
            done = False
            reward_trajectory = []
            state = tf.reshape(env.reset(), [1, -1])
            while not done:
                if render:
                    env.render()

                action, action_probability = self.act(state)
                observation, reward, done, _ = env.step(action)
                state = tf.reshape(observation, [1, -1])
                reward_trajectory.append(reward)

            rewards.append(sum(reward_trajectory))

        return rewards
