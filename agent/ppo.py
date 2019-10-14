#!/usr/bin/env python
"""Proximal Policy Optimization."""
import multiprocessing
import os
import shutil
import statistics
import time
from typing import List

import gym
import ray
import tensorflow as tf
from gym.spaces import Discrete, Box
from tensorflow.keras.optimizers import Optimizer

from agent.core import gaussian_pdf, gaussian_entropy, categorical_entropy
from agent.gather import collect, condense_worker_outputs
from utilities.util import flat_print, env_extract_dims


class PPOAgent:
    """Agent using the Proximal Policy Optimization Algorithm for learning.

    The default is an implementation using two independent models for the critic and the actor. This is of course more
    expensive than using shared parameters because we need two forward and backward calculations
    per batch however this is what is used in the original paper and most implementations. During development this also
    turned out to be beneficial for performance relative to episodes seen in easy tasks (e.g. CartPole) and crucial
    to make any significant progress in more difficult environments such as LunarLander.
    """

    def __init__(self, policy: tf.keras.Model, critic: tf.keras.Model, environment: gym.Env, horizon: int, workers: int,
                 learning_rate_pi: float, learning_rate_v: float, discount: float = 0.99, lam: float = 0.95,
                 clip: float = 0.2, c_entropy: float = 0.01):
        """Initialize the Agent.

        :param discount:                discount factor applied to future rewards
        :param clip:            clipping range for the actor's objective
        """
        super().__init__()

        self.env = environment
        self.env_name = self.env.unwrapped.spec.id
        self.state_dim, self.n_actions = env_extract_dims(self.env)

        # learning parameters
        self.horizon = horizon
        self.workers = workers
        self.discount = discount
        self.learning_rate_pi = learning_rate_pi
        self.learning_rate_v = learning_rate_v
        self.clip = tf.constant(clip)
        self.c_entropy = tf.constant(c_entropy)
        self.lam = lam

        # models and optimizers
        self.policy = policy
        self.critic = critic

        self.policy_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_pi)
        self.critic_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_v)

        # misc
        self.iteration = 0
        self.current_fps = 0
        self.device = "CPU:0"
        self.model_export_dir = "saved_models/exports/"
        os.makedirs(self.model_export_dir, exist_ok=True)

        # statistics
        self.total_frames_seen = 0
        self.episode_reward_history = []
        self.episode_length_history = []
        self.cycle_reward_history = []
        self.cycle_length_history = []
        self.entropy_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []

        # prepare for environment type
        if isinstance(self.env.action_space, Discrete):
            self.continuous_control = False
        elif isinstance(self.env.action_space, Box):
            self.continuous_control = True
        else:
            raise NotImplementedError(f"PPO cannot handle unknown Action Space Typ: {self.env.action_space}")

    def set_gpu(self, activated: bool):
        self.device = "GPU:0" if activated else "CPU:0"

    def actor_loss(self, action_prob: tf.Tensor, old_action_prob: tf.Tensor, advantage: tf.Tensor) -> tf.Tensor:
        """Actor's clipped objective as given in the PPO paper. Original objective is to be maximized
        (as given in the paper), but this is the negated objective to be minimized!

        :param action_prob:             the probability of the action for the state under the current policy
        :param old_action_prob:         the probability of the action taken given by the old policy during the episode
        :param advantage:               the advantage that taking the action gives over the estimated state value

        :return:                        the value of the objective function
        """
        r = tf.math.divide(action_prob, old_action_prob)
        return - tf.minimum(
            tf.math.multiply(r, advantage),
            tf.math.multiply(tf.clip_by_value(r, 1 - self.clip, 1 + self.clip), advantage)
        )

    def critic_loss(self, values: tf.Tensor, old_values: tf.Tensor, returns: tf.Tensor, clip=True) -> tf.Tensor:
        """Loss of the critic network as squared error between the prediction and the sampled future return.

        :param values:              value prediction by the current critic network
        :param old_values:          value prediction by the old critic network during gathering
        :param returns:             discounted return estimation

        :return:                    squared error between prediction and return
        """
        error = tf.square(values - returns)
        if clip:
            # clips value error to reduce variance
            clipped_values = old_values + tf.clip_by_value(values - old_values, -self.clip, self.clip)
            clipped_error = tf.square(clipped_values - returns)

            return tf.maximum(clipped_error, error) / 2

        return error

    def entropy_bonus(self, policy_output: tf.Tensor) -> tf.Tensor:
        """Entropy of policy output acting as regularization by preventing dominance of one action. The higher the
        entropy, the less probability mass lies on a single action, which would hinder exploration. We hence reduce
        the loss by the (scaled by c_entropy) entropy to encourage a certain degree of exploration.

        :param policy_output:   a tensor containing (batches of) probabilities for actions in the case of discrete
                                actions or (batches of) means and standard deviations for continuous control.

        :return:                (batch of) entropy bonus(es)
        """
        if self.continuous_control:
            return gaussian_entropy(stdevs=policy_output[:, self.n_actions:])
        else:
            return categorical_entropy(policy_output)

    def joint_loss_function(self, old_action_prob: tf.Tensor, action_prob: tf.Tensor, advantage: tf.Tensor,
                            prediction: tf.Tensor, discounted_return: tf.Tensor, action_probs) -> tf.Tensor:
        # loss needs to be negated since the original objective from the PPO paper is for maximization
        return self.actor_loss(action_prob, old_action_prob, advantage) \
               + self.critic_loss(prediction, discounted_return) \
               - self.c_entropy * self.entropy_bonus(action_probs)

    def drill(self, iterations: int, epochs: int, batch_size: int, worker_pool, story_teller=None):
        """Main training loop of the agent.

        Runs **iterations** cycles of experience gathering and optimization based on the gathered experience.

        :param env:             the environment on which the agent should be drilled
        :param iterations:      the number of experience-optimization cycles that shall be run
        :param epochs:          the number of epochs for which the model is optimized on the same experience data
        :param batch_size       batch size for the optimization

        :return:                self
        """
        print(f"Parallelize Over {multiprocessing.cpu_count()} Threads.\n")

        ray.init()
        for self.iteration in range(iterations):
            iteration_start = time.time()

            # run simulations in parallel
            flat_print("Gathering...")

            # export the current state of the policy and value network under unique (-enough) key
            name_key = round(time.time())
            self.policy.save(f"{self.model_export_dir}/{name_key}/policy")
            self.critic.save(f"{self.model_export_dir}/{name_key}/value")

            # parameters
            models = ray.put(f"{self.model_export_dir}/{name_key}/")
            horizon = ray.put(self.horizon)
            discount = ray.put(self.discount)
            env_name = ray.put(self.env_name)
            lam = ray.put(self.lam)

            result_object_ids = [collect.remote(models, horizon, env_name, discount, lam) for _ in range(self.workers)]
            results = [ray.get(oi) for oi in result_object_ids]
            dataset, stats = condense_worker_outputs(results)

            # clean up the saved models
            shutil.rmtree(f"{self.model_export_dir}/{name_key}")

            # process stats from actors
            self.total_frames_seen += stats.numb_processed_frames
            self.episode_length_history.extend(stats.episode_lengths)
            self.episode_reward_history.extend(stats.episode_rewards)
            self.cycle_length_history.append(None if stats.numb_completed_episodes == 0
                                             else statistics.mean(stats.episode_lengths))
            self.cycle_reward_history.append(None if stats.numb_completed_episodes == 0
                                             else statistics.mean(stats.episode_rewards))

            flat_print("Optimizing...")
            self.optimize_model(dataset, epochs, batch_size)

            iteration_end = time.time()
            self.current_fps = stats.numb_processed_frames / (iteration_end - iteration_start)

            self.report()

            if story_teller is not None and (self.iteration + 1) % story_teller.frequency == 0:
                story_teller.create_episode_gif(n=3)
            story_teller.update_graphs()
            story_teller.update_story()

        return self

    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int):
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
        actor_loss_history = []
        critic_loss_history = []
        entropy_history = []
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break bias, then divided into batches
            shuffled_dataset = dataset.shuffle(10000)  # TODO appropriate buffer size based on number of datapoints
            batched_dataset = shuffled_dataset.batch(batch_size)

            actor_epoch_losses = []
            critic_epoch_losses = []
            entropies = []
            for batch in batched_dataset:
                # use the dataset to optimize the model
                with tf.device(self.device):
                    # optimize the actor
                    with tf.GradientTape() as actor_tape:
                        policy_output = self.policy(batch["state"], training=True)

                        if self.continuous_control:
                            # if action space is continuous, calculate PDF at chosen action value
                            action_probabilities = gaussian_pdf(batch["action"],
                                                                means=policy_output[:, :self.n_actions],
                                                                stdevs=policy_output[:, self.n_actions:])
                        else:
                            # if the action space is discrete, extract the probabilities of actions actually chosen
                            action_probabilities = tf.convert_to_tensor(
                                [policy_output[i][a] for i, a in enumerate(batch["action"])])

                        # calculate the clipped loss
                        actor_loss = self.actor_loss(action_prob=action_probabilities,
                                                     old_action_prob=batch["action_prob"],
                                                     advantage=batch["advantage"])

                        entropy = self.c_entropy * self.entropy_bonus(policy_output)
                        actor_loss -= entropy
                        entropies.append(tf.reduce_mean(entropy))

                    actor_gradients = actor_tape.gradient(actor_loss, self.policy.trainable_variables)
                    self.policy_optimizer.apply_gradients(zip(actor_gradients, self.policy.trainable_variables))

                    # optimize the critic
                    with tf.GradientTape() as critic_tape:
                        old_values = batch["return"] - batch["advantage"]
                        new_values = self.critic(batch["state"], training=True)
                        critic_loss = self.critic_loss(
                            values=new_values,
                            old_values=old_values,
                            returns=batch["return"],
                        )

                    critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

                    actor_epoch_losses.append(tf.reduce_mean(actor_loss))
                    critic_epoch_losses.append(tf.reduce_mean(critic_loss))

            actor_loss_history.append(statistics.mean([numb.numpy().item() for numb in actor_epoch_losses]))
            critic_loss_history.append(statistics.mean([numb.numpy().item() for numb in critic_epoch_losses]))
            entropy_history.append(statistics.mean([numb.numpy().item() for numb in entropies]))

        self.actor_loss_history.extend(actor_loss_history)
        self.critic_loss_history.extend(critic_loss_history)
        self.entropy_history.extend(entropy_history)

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
        flat_print(f"Iteration {self.iteration:6d}: "
                   f"Mean Epi. Perf.: {0 if self.cycle_reward_history[-1] is None else round(self.cycle_reward_history[-1], 2):8.2f}; "
                   f"Mean Epi. Length: {0 if self.cycle_length_history[-1] is None else round(self.cycle_length_history[-1], 2):8.2f}; "
                   f"Total Episodes: {len(self.episode_length_history):5d}; "
                   f"Total Policy Updates: {self.policy_optimizer.iterations.numpy().item():6d}; "
                   f"Total Frames: {round(self.total_frames_seen / 1e3, 3):6.3f}k; "
                   f"Exec. Speed: {self.current_fps:6.2f}fps\n")
