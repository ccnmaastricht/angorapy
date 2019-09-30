#!/usr/bin/env python
"""Proximal Policy Optimization Implementation."""
import statistics
from abc import abstractmethod
from typing import Tuple, List

import gym
import tensorflow as tf

from agent.core import _RLAgent
from agent.gather import _Gatherer
from util import flat_print
from visualization.story import StoryTeller


class _PPOBase(_RLAgent):
    """Agent using the Proximal Policy Optimization Algorithm for learning."""

    def __init__(self, gatherer: _Gatherer, learning_rate: float, discount: float = 0.99, epsilon_clip: float = 0.2,
                 c_entropy: float = 0.01):
        """Initialize the Agent.

        :param learning_rate:           the agents learning rate
        :param discount:                discount factor applied to future rewards
        :param epsilon_clip:            clipping range for the actor's objective
        """
        super().__init__()

        self.gatherer = gatherer

        # learning parameters
        self.discount = tf.constant(discount, dtype=tf.float64)
        self.learning_rate = learning_rate
        self.epsilon_clip = tf.constant(epsilon_clip, dtype=tf.float64)
        self.c_entropy = tf.constant(c_entropy, dtype=tf.float64)

        # misc
        self.iteration = 0
        self.device = "CPU:0"

        self.policy_optimizer: tf.keras.optimizers.Optimizer = None

    @abstractmethod
    def full_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        pass

    @abstractmethod
    def critic_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        pass

    @abstractmethod
    def actor_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        pass

    @abstractmethod
    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int):
        pass

    def set_gpu(self, activated: bool) -> None:
        self.device = "GPU:0" if activated else "CPU:0"

    def act(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        probabilities = self.actor_prediction(state)
        action = tf.random.categorical(tf.math.log(probabilities), 1)[0][0]

        return action, probabilities[0][action]

    # LOSS

    def actor_loss(self, action_prob: tf.Tensor, old_action_prob: tf.Tensor, advantage: tf.Tensor) -> tf.Tensor:
        """Actor's clipped objective as given in the PPO paper. Original objective is to be maximized
        (as given in the paper), but this is the negated objective to be minimized!

        :param old_action_prob:         the probability of the action taken given by the old policy during the episode
        :param action_prob:             the probability of the action for the state under the current policy
                                        (from here the gradients go backwards)
        :param advantage:               the advantage that taking the action gives over the estimated state value

        :return:                        the value of the objective function
        """
        r = tf.math.divide(action_prob, old_action_prob)
        return - tf.minimum(
            tf.math.multiply(r, advantage),
            tf.math.multiply(tf.clip_by_value(r, 1 - self.epsilon_clip, 1 + self.epsilon_clip), advantage)
        )

    @staticmethod
    def critic_loss(prediction: tf.Tensor, v_GAE: tf.Tensor) -> tf.Tensor:
        """Loss of the critic network as squared error between the prediction and the sampled future return.

        :param prediction:      prediction that the critic network made
        :param v_GAE:           discounted return estimated by GAE

        :return:                squared error between prediction and return
        """
        return tf.square(prediction - v_GAE)

    @staticmethod
    def entropy_bonus(action_probs):
        """Entropy of policy output acting as regularization by preventing dominance of one action."""
        return - tf.reduce_sum(action_probs * tf.math.log(action_probs), 1)

    def joint_loss_function(self, old_action_prob: tf.Tensor, action_prob: tf.Tensor, advantage: tf.Tensor,
                            prediction: tf.Tensor, discounted_return: tf.Tensor, action_probs) -> tf.Tensor:
        # loss needs to be negated since the original objective from the PPO paper is for maximization
        return self.actor_loss(action_prob, old_action_prob, advantage) \
               + self.critic_loss(prediction, discounted_return) \
               - self.c_entropy * self.entropy_bonus(action_probs)

    # TRAIN

    def drill(self, env: gym.Env, iterations: int, epochs: int, batch_size: int, story_teller: StoryTeller = None):
        """Main training loop of the agent.

        Runs **iterations** cycles of experience gathering and optimization based on the gathered experience.

        :param update_story_every:
        :param env:             the environment on which the agent should be drilled
        :param iterations:      the number of experience-optimization cycles that shall be run
        :param epochs:          the number of epochs for which the model is optimized on the same experience data
        :param agents:          the number of trajectories generated during the experience gathering
        :param batch_size       batch size for the optimization

        :return:                self
        """
        for self.iteration in range(iterations):
            # run simulations
            flat_print("Gathering...")
            dataset = self.gatherer.gather(self)

            flat_print("Optimizing...")
            self.optimize_model(dataset, epochs, batch_size)

            self.report()

            if story_teller is not None and (self.iteration + 1) % story_teller.frequency == 0:
                story_teller.update_reward_graph()
                story_teller.create_episode_gif(n=3)
                story_teller.update_story()

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

    def report(self):
        mean_perf = statistics.mean(self.gatherer.episode_reward_history[-self.gatherer.last_episodes_completed:])
        mean_episode_length = statistics.mean(
            self.gatherer.episode_length_history[-self.gatherer.last_episodes_completed:])

        flat_print(f"Iteration {self.iteration:6d}: "
                   f"Mean Epi. Perf.: {round(mean_perf, 2):8.2f}; "
                   f"Mean Epi. Length: {round(mean_episode_length, 2):8.2f}; "
                   f"It. Steps: {self.gatherer.steps_during_last_gather:6d}; "
                   f"Total Policy Updates: {self.policy_optimizer.iterations.numpy().item()}; "
                   f"Total Frames: {round(self.gatherer.total_frames / 1e3, 3)}k\n")


@DeprecationWarning
class PPOAgentJoint(_PPOBase):
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


class PPOAgentDual(_PPOBase):
    """PPO implementation using two independent models for the critic and the actor.

    This is of course more expensive than using shared parameters because we need two forward and backward calculations
    per batch however this is what is used in the original paper and most implementations. During development this also
    turned out to be beneficial for performance relative to episodes seen in easy tasks (e.g. CartPole) and crucial
    to make any significant progress in more difficult environments such as LunarLander.
    """

    def __init__(self, policy: tf.keras.Model, critic: tf.keras.Model, gatherer: _Gatherer, learning_rate: float,
                 discount: float, epsilon_clip: float, c_entropy: float):
        super().__init__(gatherer, learning_rate=learning_rate, discount=discount, epsilon_clip=epsilon_clip,
                         c_entropy=c_entropy)

        self.policy = policy
        self.critic = critic

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def full_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.policy(state, training=training), self.critic(state, training=training)

    def critic_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.critic(state, training=training)

    def actor_prediction(self, state, training=False):
        """Wrapper to allow for shared and non-shared models."""
        return self.policy(state, training=training)

    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> List[Tuple[int, int]]:
        """Optimize the agent's policy and value network based on a given dataset.

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
        loss_history = []
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break bias, then divided into batches
            shuffled_dataset = dataset.shuffle(10000)  # TODO appropriate buffer size based on number of datapoints
            batched_dataset = shuffled_dataset.batch(batch_size)

            epoch_losses = []
            for batch in batched_dataset:
                # use the dataset to optimize the model
                with tf.device(self.device):
                    # optimize the actor
                    with tf.GradientTape() as actor_tape:
                        action_probabilities = self.actor_prediction(batch["state"], training=True)
                        chosen_action_probabilities = tf.convert_to_tensor(
                            [action_probabilities[i][a] for i, a in enumerate(batch["action"])], dtype=tf.float64)

                        actor_loss = self.actor_loss(action_prob=chosen_action_probabilities,
                                                     old_action_prob=batch["action_prob"],
                                                     advantage=batch["advantage"]) \
                                     + self.c_entropy * self.entropy_bonus(action_probabilities)

                    actor_gradients = actor_tape.gradient(actor_loss, self.policy.trainable_variables)
                    self.policy_optimizer.apply_gradients(zip(actor_gradients, self.policy.trainable_variables))

                    # optimize the critic
                    with tf.GradientTape() as critic_tape:
                        critic_loss = self.critic_loss(prediction=self.critic_prediction(batch["state"], training=True),
                                                       v_GAE=batch["return"])

                    critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

                    epoch_losses.append([tf.reduce_mean(actor_loss), tf.reduce_mean(critic_loss)])

            loss_history.append(
                tf.reduce_mean(epoch_losses, 0).numpy().round(2).tolist()
            )

        return loss_history
