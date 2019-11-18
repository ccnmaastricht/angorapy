#!/usr/bin/env python
"""Implementation of Proximal Policy Optimization Algorithm."""
import json
import multiprocessing
import os
import re
import shutil
import statistics
import time
from collections import OrderedDict
from inspect import getfullargspec as fargs
from typing import List, Tuple

import gym
import numpy
import ray
import tensorflow as tf
from gym.spaces import Discrete, Box
from tensorflow.keras.optimizers import Optimizer

from agent.core import gaussian_pdf, gaussian_entropy, categorical_entropy
from agent.gather import collect, \
    read_dataset_from_storage, condense_stats
from agent.policy import act_discrete, act_continuous
from utilities.const import COLORS, BASE_SAVE_PATH
from utilities.datatypes import ModelTuple
from utilities.util import flat_print, env_extract_dims, parse_state, add_state_dims, merge_into_batch, \
    is_recurrent_model


class PPOAgent:
    """Agent using the Proximal Policy Optimization Algorithm for learning.
    
    The default is an implementation using two independent models for the critic and the actor. This is of course more
    expensive than using shared parameters because we need two forward and backward calculations
    per batch however this is what is used in the original paper and most implementations. During development this also
    turned out to be beneficial for performance relative to episodes seen in easy tasks (e.g. CartPole) and crucial
    to make any significant progress in more difficult environments such as LunarLander.
    """
    policy: tf.keras.Model
    value: tf.keras.Model
    joint: tf.keras.Model

    def __init__(self, model_builder, environment: gym.Env, horizon: int, workers: int, learning_rate_pi: float = 0.001,
                 learning_rate_v: float = 0.001, discount: float = 0.99, lam: float = 0.95, clip: float = 0.2,
                 c_entropy: float = 0.01, c_value: float = 0.5, gradient_clipping: float = None,
                 clip_values: bool = True, _make_dirs=True, debug: bool = False):
        super().__init__()
        self.debug = debug

        # environment info
        self.env = environment
        self.env_name = self.env.unwrapped.spec.id
        self.state_dim, self.n_actions = env_extract_dims(self.env)
        if isinstance(self.env.action_space, Discrete):
            self.continuous_control = False
        elif isinstance(self.env.action_space, Box):
            self.continuous_control = True
        else:
            raise NotImplementedError(f"PPO cannot handle unknown Action Space Typ: {self.env.action_space}")

        # learning parameters
        self.horizon = horizon
        self.workers = workers
        self.discount = discount
        self.learning_rate_pi = learning_rate_pi
        self.learning_rate_v = learning_rate_v
        self.clip = clip
        self.c_entropy = c_entropy
        self.c_value = c_value
        self.lam = lam
        self.gradient_clipping = gradient_clipping
        self.clip_values = clip_values

        # models and optimizers
        self.model_builder = model_builder
        self.builder_function_name = model_builder.__name__
        self.policy, self.value, self.joint = model_builder(self.env, **({"bs": 1} if "bs" in fargs(
            model_builder).args else {}))
        self.optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_pi, epsilon=1e-5)

        # passing one sample, which for some reason prevents cuDNN init error
        if "observation" in self.env.observation_space.sample():
            self.joint(merge_into_batch(
                [add_state_dims(self.env.observation_space.sample()["observation"], dims=1) for _ in range(1)]))

        # miscellaneous
        self.iteration = 0
        self.current_fps = 0
        self.device = "CPU:0"
        self.model_export_dir = "saved_models/exports/"
        self.agent_id = round(time.time())
        self.agent_directory = f"{BASE_SAVE_PATH}/{self.agent_id}/"
        if _make_dirs:
            os.makedirs(self.model_export_dir, exist_ok=True)
            os.makedirs(self.agent_directory)

        shutil.rmtree("storage/experience/")
        os.makedirs("storage/experience/", exist_ok=True)

        # statistics
        self.total_frames_seen = 0
        self.total_episodes_seen = 0
        self.episode_reward_history = []
        self.episode_length_history = []
        self.cycle_reward_history = []
        self.cycle_length_history = []
        self.entropy_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.time_dicts = []

    def set_gpu(self, activated: bool):
        """Set GPU usage mode."""
        self.device = "GPU:0" if activated else "CPU:0"

    def policy_loss(self, action_prob: tf.Tensor, old_action_prob: tf.Tensor, advantage: tf.Tensor) -> tf.Tensor:
        """Actor's clipped objective as given in the PPO paper. Original objective is to be maximized
        (as given in the paper), but this is the negated objective to be minimized!

        Args:
          action_prob (tf.Tensor): the probability of the action for the state under the current policy
          old_action_prob (tf.Tensor): the probability of the action taken given by the old policy during the episode
          advantage (tf.Tensor): the advantage that taking the action gives over the estimated state value

        Returns:
          the value of the objective function

        """
        r = action_prob / old_action_prob
        return tf.reduce_mean(tf.maximum(
            - tf.math.multiply(r, advantage),
            - tf.math.multiply(tf.clip_by_value(r, 1 - self.clip, 1 + self.clip), advantage)
        ))

    def value_loss(self, value_predictions: tf.Tensor, old_values: tf.Tensor, returns: tf.Tensor,
                   clip: bool = True) -> tf.Tensor:
        """Loss of the critic network as squared error between the prediction and the sampled future return.

        Args:
          value_predictions (tf.Tensor): value prediction by the current critic network
          old_values (tf.Tensor): value prediction by the old critic network during gathering
          returns (tf.Tensor): discounted return estimation
          clip (object): (Default value = True) value loss can be clipped by same range as policy loss

        Returns:
          squared error between prediction and return

        """
        error = tf.square(value_predictions - returns)
        if clip:
            # clips value error to reduce variance
            clipped_values = old_values + tf.clip_by_value(value_predictions - old_values, -self.clip, self.clip)
            clipped_error = tf.square(clipped_values - returns)

            return tf.maximum(clipped_error, error) * 0.5

        return tf.reduce_mean(error)

    def entropy_bonus(self, policy_output: tf.Tensor) -> tf.Tensor:
        """Entropy of policy output acting as regularization by preventing dominance of one action. The higher the
        entropy, the less probability mass lies on a single action, which would hinder exploration. We hence reduce
        the loss by the (scaled by c_entropy) entropy to encourage a certain degree of exploration.

        Args:
          policy_output (tf.Tensor): a tensor containing (batches of) probabilities for actions in the case of discrete
            actions or (batches of) means and standard deviations for continuous control.

        Returns:
          entropy bonus
        """
        if self.continuous_control:
            return tf.reduce_mean(gaussian_entropy(stdevs=policy_output[:, self.n_actions:]))
        else:
            return tf.reduce_mean(categorical_entropy(policy_output))

    def drill(self, n: int, epochs: int, batch_size: int, story_teller=None, export: bool = False, save_every: int = 0,
              separate_eval: bool = False) -> "PPOAgent":
        """Start a training loop of the agent.
        
        Runs **n** cycles of experience gathering and optimization based on the gathered experience.

        Args:
            n (int): the number of experience-optimization cycles that shall be run
            epochs (int): the number of epochs for which the model is optimized on the same experience data
            batch_size (int): batch size for the optimization
            story_teller: story telling object that creates visualizations of the training process on the fly (Default
                value = None)
            export (bool): boolean indicator for whether communication with workers is achieved through file saving
                or direct weight passing (Default value = False)
            save_every (int): for any int x > 0 save the policy every x iterations, if x = 0 (default) do not save
            separate_eval (bool): if false (default), use episodes from gathering for statistics, if true, evaluate 10
                additional episodes.

        Returns:
            self
        """
        assert self.horizon * self.workers >= batch_size, "Batch Size is larger than the number of transitions."

        ray.init(local_mode=self.debug)

        # rebuild model with desired batch size
        weights = self.joint.get_weights()
        self.policy, self.value, self.joint = self.model_builder(self.env, **({"bs": batch_size} if "bs" in fargs(
            self.model_builder).args else {}))
        self.joint.set_weights(weights)

        print(f"Parallelizing {self.workers} Workers Over {multiprocessing.cpu_count()} Threads.\n")
        for self.iteration in range(self.iteration, n):
            time_dict = OrderedDict()
            subprocess_start = time.time()

            # run simulations in parallel
            flat_print("Gathering...")

            # export the current state of the policy and value network under unique (-enough) key
            name_key = round(time.time())
            if export:
                self.policy.save(f"{self.model_export_dir}/{name_key}/policy")
                self.value.save(f"{self.model_export_dir}/{name_key}/value")
                models = f"{self.model_export_dir}/{name_key}/"
            else:
                policy_tuple = ModelTuple(self.builder_function_name, self.policy.get_weights())
                critic_tuple = ModelTuple(self.builder_function_name, self.value.get_weights())
                models = (policy_tuple, critic_tuple)

            # create processes and execute them
            result_ids = [collect.remote(models, self.horizon, self.env_name, self.discount, self.lam, pid) for pid in
                          range(self.workers)]
            split_stats = [ray.get(oi) for oi in result_ids]
            stats = condense_stats(split_stats)

            time_dict["gathering"] = time.time() - subprocess_start
            subprocess_start = time.time()

            dataset = read_dataset_from_storage(dtype_actions=tf.float32 if self.continuous_control else tf.int32,
                                                is_shadow_hand=isinstance(self.state_dim, tuple))

            time_dict["communication"] = time.time() - subprocess_start
            subprocess_start = time.time()

            # clean up the saved models
            if export:
                shutil.rmtree(f"{self.model_export_dir}/{name_key}")

            # process stats from actors
            if not separate_eval:
                if stats.numb_completed_episodes == 0:
                    print("WARNING: You are using a horizon that caused this cycle to not finish a single episode. "
                          "Consider activating seperate evaluation in drill() to get meaningful statistics.")

                self.episode_length_history.extend(stats.episode_lengths)
                self.episode_reward_history.extend(stats.episode_rewards)
                self.cycle_length_history.append(None if stats.numb_completed_episodes == 0
                                                 else statistics.mean(stats.episode_lengths))
                self.cycle_reward_history.append(None if stats.numb_completed_episodes == 0
                                                 else statistics.mean(stats.episode_rewards))
            else:
                flat_print("Evaluating...")
                eval_lengths, eval_rewards = self.evaluate(10)
                self.episode_length_history.extend(eval_lengths)
                self.episode_reward_history.extend(eval_rewards)
                self.cycle_length_history.append(statistics.mean(eval_lengths))
                self.cycle_reward_history.append(statistics.mean(eval_rewards))

            time_dict["evaluating"] = time.time() - subprocess_start
            subprocess_start = time.time()

            self.report()

            flat_print("Optimizing...")
            self.optimize_model(dataset, epochs, batch_size)

            time_dict["optimizing"] = time.time() - subprocess_start
            subprocess_start = time.time()

            flat_print("Finalizing...")
            self.time_dicts.append(time_dict)
            self.current_fps = stats.numb_processed_frames / (sum([v for v in time_dict.values() if v is not None]))
            self.total_frames_seen += stats.numb_processed_frames
            self.total_episodes_seen += stats.numb_completed_episodes

            if story_teller is not None and story_teller.frequency != 0 and (
                    self.iteration + 1) % story_teller.frequency == 0:
                print("Creating Episode GIFs for current state of policy...")
                story_teller.create_episode_gif(n=3)
            story_teller.update()

            if save_every != 0 and self.iteration != 0 and (self.iteration + 1) % save_every == 0:
                print("Saving the current state of the agent.")
                self.save_agent_state()

            time_dict["finalizing"] = time.time() - subprocess_start

        return self

    def optimize_model(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> None:
        """Optimize the agent's policy and value network based on a given dataset.
        
        Since data processing is apparently not possible with tensorflow data sets on a GPU, we will only let the GPU
        handle the training, but keep the rest of the data pipeline on the CPU. I am not currently sure if this is the
        best approach, but it might be the good anyways for large data chunks anyways due to GPU memory limits. It also
        should not make much of a difference since gym runs entirely on CPU anyways, hence for every experience
        gathering we need to transfer all Tensors from CPU to GPU, no matter whether the dataset is stored on GPU or
        not. Even more so this applies with running simulations on the cluster.

        Args:
            dataset (tf.data.Dataset): tensorflow dataset containing s, a, p(a), r and A as components per data point
            epochs (int): number of epochs to train on this dataset
            batch_size (int): batch size with which the dataset is sampled

        Returns:
            None
        """
        actor_loss_history, critic_loss_history, entropy_history = [], [], []
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break correlation, then divided into batches
            # shuffled_dataset = dataset.shuffle(10000)  # TODO appropriate shuffling sensitive to stateful orderedness
            batched_dataset = dataset.batch(batch_size, drop_remainder=True)

            actor_epoch_losses, value_epoch_losses, entropies = [], [], []
            for batch in batched_dataset:

                # use the dataset to optimize the model
                with tf.device(self.device):

                    # optimize policy and value network simultaneously
                    with tf.GradientTape() as tape:
                        state_batch = batch["state"] if "state" in batch else (batch["in_vision"], batch["in_proprio"],
                                                                               batch["in_touch"], batch["in_goal"])
                        policy_output, value_output = self.joint(state_batch, training=True)
                        old_values = batch["return"] - batch["advantage"]

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
                        policy_loss = self.policy_loss(action_prob=action_probabilities,
                                                       old_action_prob=batch["action_prob"],
                                                       advantage=batch["advantage"])
                        value_loss = self.value_loss(value_predictions=value_output,
                                                     old_values=old_values,
                                                     returns=batch["return"],
                                                     clip=self.clip_values)
                        entropy = self.entropy_bonus(policy_output)
                        total_loss = policy_loss + tf.multiply(self.c_value, value_loss) - tf.multiply(self.c_entropy,
                                                                                                       entropy)

                    gradients = tape.gradient(total_loss, self.joint.trainable_variables)
                    if self.gradient_clipping is not None:
                        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
                    self.optimizer.apply_gradients(zip(gradients, self.joint.trainable_variables))

                    entropies.append(tf.reduce_mean(entropy))
                    actor_epoch_losses.append(tf.reduce_mean(policy_loss))
                    value_epoch_losses.append(tf.reduce_mean(value_loss))

            # reset RNN states after an epoch if there are any
            self.joint.reset_states()

            # remember some statistics
            actor_loss_history.append(statistics.mean([numb.numpy().item() for numb in actor_epoch_losses]))
            critic_loss_history.append(statistics.mean([numb.numpy().item() for numb in value_epoch_losses]))
            entropy_history.append(statistics.mean([numb.numpy().item() for numb in entropies]))

        # store statistics in agent history
        self.actor_loss_history.extend(actor_loss_history)
        self.critic_loss_history.extend(critic_loss_history)
        self.entropy_history.extend(entropy_history)

    def evaluate(self, n: int, render=False) -> Tuple[List[int], List[int]]:
        """Evaluate the current state of the policy on the given environment for n episodes. Optionally can render to
        visually inspect the performance.

        Args:
            n (int): integer value indicating the number of episodes that shall be run
            render (bool): whether to render it

        Returns:
            two lists of length n, giving episode lengths and rewards respectively
        """
        rewards, lengths = [], []
        is_recurrent = is_recurrent_model(self.policy)
        policy_act = act_discrete if not self.continuous_control else act_continuous
        for episode in range(n):
            done = False
            reward_trajectory = []
            length = 0
            state = parse_state(self.env.reset())
            while not done:
                action, action_prob = policy_act(self.policy, add_state_dims(state, dims=2 if is_recurrent else 1))
                self.env.render() if render else None
                observation, reward, done, _ = self.env.step(action)
                state = parse_state(self.env.reset())
                reward_trajectory.append(reward)
                length += 1

            lengths.append(length)
            rewards.append(sum(reward_trajectory))

        return lengths, rewards

    def report(self):
        """Print a report of the current state of the training."""
        sc, nc, ec = COLORS["OKGREEN"], COLORS["OKBLUE"], COLORS["ENDC"]
        time_distribution_string = ""
        if len(self.time_dicts) > 0:
            times = [time for time in self.time_dicts[-1].values() if time is not None]
            jobs = [job[0] for job, time in self.time_dicts[-1].items() if time is not None]
            time_percentages = [str(round(100 * t / sum(times))) + jobs[i] for i, t in enumerate(times)]
            time_distribution_string = "[" + "|".join(map(str, time_percentages)) + "]"
        flat_print(f"{sc}{f'Iteration {self.iteration:5d}' if self.iteration != 0 else 'Before Training'}{ec}: "
                   f"AvgRew.: {nc}{0 if self.cycle_reward_history[-1] is None else round(self.cycle_reward_history[-1], 2):8.2f}{ec}; "
                   f"AvgLen.: {nc}{0 if self.cycle_length_history[-1] is None else round(self.cycle_length_history[-1], 2):8.2f}{ec}; "
                   f"AvgEnt.: {nc}{0 if len(self.entropy_history) == 0 else round(self.entropy_history[-1], 2):5.2f}{ec}; "
                   f"Eps.: {nc}{self.total_episodes_seen:5d}{ec}; "
                   f"Updates: {nc}{self.optimizer.iterations.numpy().item():6d}{ec}; "
                   f"Frames: {nc}{round(self.total_frames_seen / 1e3, 3):8.3f}{ec}k; "
                   f"Speed: {nc}{self.current_fps:7.2f}{ec}fps {time_distribution_string}\n")

    def save_agent_state(self):
        """Save the current state of the agent into the agent directory, identified by the current iteration."""
        self.policy.save(self.agent_directory + f"/{self.iteration}/policy")
        self.value.save(self.agent_directory + f"/{self.iteration}/value")

        with open(self.agent_directory + f"/{self.iteration}/parameters.json", "w") as f:
            json.dump(self.get_parameters(), f)

    def get_parameters(self):
        """Get the agents parameters necessary to reconstruct it."""
        parameters = self.__dict__.copy()
        del parameters["env"]
        del parameters["policy"], parameters["value"], parameters["policy_value"]
        del parameters["policy_optimizer"], parameters["value_optimizer"]

        return parameters

    @staticmethod
    def from_agent_state(agent_id: int) -> "PPOAgent":
        """Build an agent from a previously saved state.

        Args:
            agent_id: the ID of the agent to be loaded

        Returns:
            loaded_agent: a PPOAgent object of the same state as the one saved into the path specified by agent_id
        """
        # TODO also load the state of the optimizers
        agent_path = BASE_SAVE_PATH + f"/{agent_id}"
        if not os.path.isdir(agent_path):
            raise FileNotFoundError("The given agent ID does not match any existing save history.")

        if len(os.listdir(agent_path)) == 0:
            raise FileNotFoundError("The given agent ID's save history is empty.")

        latest = max([int(re.match("([0-9]+)", fn).group(0)) for fn in os.listdir(agent_path)])
        policy = tf.keras.models.load_model(f"{agent_path}/{latest}/policy")
        value = tf.keras.models.load_model(f"{agent_path}/{latest}/value")

        with open(f"{agent_path}/{latest}/parameters.json", "r") as f:
            parameters = json.load(f)

        env = gym.make(parameters["env_name"])
        example_input = env.reset().reshape([1, -1]).astype(numpy.float32)
        value.predict(example_input)

        loaded_agent = PPOAgent(policy, value, env, parameters["horizon"], parameters["workers"], _make_dirs=False)

        for p, v in parameters.items():
            loaded_agent.__dict__[p] = v

        return loaded_agent
