#!/usr/bin/env python
"""TODO Module Docstring."""
import itertools
from typing import List, Tuple

import numpy
import tensorflow as tf

from agent.gather import Gatherer
from agent.ppo import PPOAgent


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