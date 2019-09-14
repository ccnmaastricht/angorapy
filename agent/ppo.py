#!/usr/bin/env python
"""Proximal Policy Optimization Implementation."""
import itertools
import statistics
from typing import Tuple, List

import gym
import tensorflow as tf

from agent.core import _RLAgent, get_discounted_returns
from policy_networks.fully_connected import PPOActorCriticNetwork, PPOCriticNetwork, PPOActorNetwork


class PPOAgent(_RLAgent):
    """Agent using the Proximal Policy Optimization Algorithm for learning."""

    def __init__(self, state_dimensionality, n_actions, learning_rate: float, discount: float, epsilon_clip: float):
        """Initialize the Agent.

        :param learning_rate:           the agents learning rate
        :param discount:                discount factor applied to future rewards
        :param epsilon_clip:            clipping range for the actor's objective
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
        self.entropy_coefficient = tf.constant(0.01, dtype=tf.float64)

        # Models
        self._build_models()

        self.device = "CPU:0"

    def _build_models(self):
        self.model = PPOActorCriticNetwork(self.state_dimensionality, self.n_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def set_gpu(self, activated: bool) -> None:
        self.device = "GPU:0" if activated else "CPU:0"

    def act(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        probabilities = self.actor_prediction(state)
        action = tf.random.categorical(tf.math.log(probabilities), 1)[0][0]

        return action, probabilities[0][action]

    def full_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.model(state, training=training)

    def critic_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.model(state, training=training)[1]

    def actor_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.model(state, training=training)[0]

    # OBJECTIVES

    def _actor_objective(self, old_action_prob: tf.Tensor, action_prob: tf.Tensor, advantage: tf.Tensor) -> tf.Tensor:
        """Actor's clipped objective as given in the PPO paper.
        The objective is to be maximized (as given in the paper)!

        :param old_action_prob:         the probability of the action taken given by the old policy during the episode
        :param action_prob:             the probability of the action for the state under the current policy
                                        (from here the gradients go backwards)
        :param advantage:               the advantage that taking the action gives over the estimated state value

        :return:                        the value of the objective function
        """
        r = tf.math.divide(action_prob, old_action_prob)
        return tf.minimum(
            tf.math.multiply(r, advantage),
            tf.math.multiply(tf.clip_by_value(r, 1 - self.epsilon_clip, 1 + self.epsilon_clip), advantage)
        )

    @staticmethod
    def _critic_loss(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """Loss of the critic network as squared error between the prediction and the sampled future return.

        :param prediction:      prediction that the critic network made
        :param target:          discounted return

        :return:                squared error between prediction and return
        """
        return tf.square(prediction - target)

    @staticmethod
    def entropy_bonus(action_probs):
        """Entropy of policy output acting as regularization by preventing dominance of on action."""
        return -tf.reduce_sum(action_probs * tf.log(action_probs), 1)

    def joint_loss_function(self, old_action_prob: tf.Tensor, action_prob: tf.Tensor, advantage: tf.Tensor,
                            prediction: tf.Tensor, discounted_return: tf.Tensor, action_probs):
        # loss needs to be negated since the original objective from the PPO paper is for maximization
        return - (self._actor_objective(old_action_prob, action_prob, advantage)
                  - self._critic_loss(prediction, discounted_return)
                  + self.entropy_coefficient * self.entropy_bonus(action_probs))

    def gather_experience(self, env: gym.Env, n_trajectories: int) -> Tuple[List, List, List, List]:
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
                observation, reward, done, _ = env.step(action.numpy())

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

            print(f"Iteration {iteration}: Average reward of over {len(r_trajectories)} episodes: "
                  f"{statistics.mean([sum(r) for r in r_trajectories])}")
            if iteration > 10:
                self.evaluate(env, 1, True)

            discounted_returns = [get_discounted_returns(reward_trajectory, self.discount) for reward_trajectory in
                                  r_trajectories]
            state_value_predictions = [[self.critic_prediction(tf.reshape(state, [1, -1]))[0][0] for state in trajectory]
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

    def evaluate(self, env: gym.Env, n: int, render: bool = False) -> List[int]:
        """Evaluate the current state of the policy on the given environment for n episodes. Optionally can render to
        visually inspect the performance.

        :param env:         a gym environment
        :param n:           integer value indicating the number of episodes that shall be run
        :param render:      whether to render the episodes or not

        :return:            a list of length n of episode rewards
        """
        rewards = []
        for episode in range(n):
            done = False
            reward_trajectory = []
            state = tf.reshape(env.reset(), [1, -1])
            while not done:
                if render:
                    env.render()

                action, action_probability = self.act(state)
                observation, reward, done, _ = env.step(action.numpy())
                state = tf.reshape(observation, [1, -1])
                reward_trajectory.append(reward)

            rewards.append(sum(reward_trajectory))

        return rewards


class PPOAgentDual(PPOAgent):
    """PPO implementation using two independent models for the critic and the actor.

    This is of course more expensive than using shared parameters because we need two forward and backward calculations
    per batch however this is what is used in the original paper and most implementations. During development this also
    turned out to be beneficial for performance relative to episodes seen in easy tasks (e.g. CartPole) and crucial
    to make any significant progress in more difficult environments such as LunarLander.
    """

    def __init__(self, state_dimensionality, n_actions, learning_rate: float, discount: float, epsilon_clip: float):
        """Initialize the Agent.

        :param learning_rate:           the agents learning rate
        :param discount:                discount factor applied to future rewards
        :param epsilon_clip:            clipping range for the actor's objective
        :param state_dimensionality:    number of dimensions in the states that the agent has to process
        :param n_actions:               number of actions the agent can choose from
        """
        super().__init__(state_dimensionality, n_actions, learning_rate, discount, epsilon_clip)

    def _build_models(self):
        self.actor_model = PPOActorNetwork(self.state_dimensionality, self.n_actions)
        self.critic_model = PPOCriticNetwork(self.state_dimensionality, self.n_actions)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def full_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.actor_model(state, training=training), self.critic_model(state, training=training)

    def critic_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.critic_model(state, training=training)

    def actor_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.actor_model(state, training=training)

    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> None:
        """Optimize the agent's policy and value network based on a given dataset.

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
                with tf.device(self.device):
                    # optimize the actor
                    with tf.GradientTape() as tape:
                        action_probabilities = self.actor_prediction(batch["state"], training=True)
                        chosen_action_probabilities = tf.convert_to_tensor(
                            [action_probabilities[i][a] for i, a in enumerate(batch["action"])], dtype=tf.float64)

                        actor_loss = - self._actor_objective(
                            old_action_prob=batch["action_prob"],
                            action_prob=chosen_action_probabilities,
                            advantage=batch["advantage"])

                    actor_gradients = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))

                    # optimize the critic
                    with tf.GradientTape() as tape:
                        critic_loss = self._critic_loss(
                            prediction=self.critic_prediction(batch["state"], training=True),
                            target=batch["return"])

                    critic_gradients = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_model.trainable_variables))
