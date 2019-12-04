#!/usr/bin/env python
"""Implementation of Proximal Policy Optimization Algorithm."""
import json
import logging
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
import ray
import tensorflow as tf
from gym.spaces import Discrete, Box, Dict
from tensorflow.keras.optimizers import Optimizer
from tqdm import tqdm

import models
from agent.core import gaussian_pdf, gaussian_entropy, categorical_entropy, extract_discrete_action_probabilities
from agent.dataio import read_dataset_from_storage
from agent.gather import collect, evaluate
from utilities.const import COLORS, BASE_SAVE_PATH
from utilities.datatypes import ModelTuple, condense_stats
from utilities.util import flat_print, env_extract_dims, add_state_dims, merge_into_batch, is_recurrent_model


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

    def __init__(self, model_builder, environment: gym.Env, horizon: int, workers: int, learning_rate: float = 0.001,
                 discount: float = 0.99, lam: float = 0.95, clip: float = 0.2,
                 c_entropy: float = 0.01, c_value: float = 0.5, gradient_clipping: float = None,
                 clip_values: bool = True, tbptt_length: int = 16, lr_schedule: str = None,
                 _make_dirs=True, debug: bool = False):
        """ Initialize the PPOAgent with given hyperparameters. Policy and value network will be freshly initialized.

        Args:
            model_builder: a function creating a policy, value and joint model
            environment (gym.Env): the environment in which the agent will learn 
            horizon (int): the number of timesteps each worker collects 
            workers (int): the number of workers
            learning_rate (float): the learning rate of the Adam optimizer
            discount (float): discount factor for future rewards during collection
            lam (float): lambda parameter of the generalized advantage estimation
            clip (float): clipping range for both policy and value loss
            c_entropy (float): coefficient for entropy in the combined loss
            c_value (float): coefficient fot value in the combined loss
            gradient_clipping (float): max norm for the gradient clipping, set None to deactivate (default)
            clip_values (bool): boolean switch to turn off or on the clipping of the value loss
            lr_schedule (str): type of scheduler for the learning rate, default None (constant learning rate). Can be
                either None or 'exponential'
            _make_dirs (bool): internal parameter to indicate whether or not to recreate the directories
            debug (bool): turn on/off debugging mode
        """
        super().__init__()
        self.debug = debug

        # checkups
        assert lr_schedule is None or isinstance(lr_schedule, str)

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

        # hyperparameters
        self.horizon = horizon
        self.workers = workers
        self.discount = discount
        self.learning_rate = learning_rate
        self.clip = clip
        self.c_entropy = tf.constant(c_entropy, dtype=tf.float32)
        self.c_value = tf.constant(c_value, dtype=tf.float32)
        self.lam = lam
        self.gradient_clipping = gradient_clipping if gradient_clipping != 0 else None
        self.clip_values = clip_values
        self.tbptt_length = tbptt_length

        # learning rate schedule
        if lr_schedule is None:
            self.lr_schedule = learning_rate
        elif lr_schedule.lower() == "exponential":
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=1000,
                decay_rate=0.98
            )
        else:
            raise ValueError("Unknown Schedule type. Choose one of (None, exponential)")

        # models and optimizers
        self.model_builder = model_builder
        self.builder_function_name = model_builder.__name__
        self.policy, self.value, self.joint = model_builder(self.env, **({"bs": 1} if "bs" in fargs(
            model_builder).args else {}))
        self.optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-5)
        self.is_recurrent = is_recurrent_model(self.policy)
        if not self.is_recurrent:
            self.tbptt_length = 1

        # passing one sample, which for some reason prevents cuDNN init error
        if isinstance(self.env.observation_space, Dict) and "observation" in self.env.observation_space.sample():
            self.joint(merge_into_batch(
                [add_state_dims(self.env.observation_space.sample()["observation"], dims=1) for _ in range(1)]))

        # miscellaneous
        self.iteration = 0
        self.current_fps = 0
        self.device = "CPU:0"
        self.model_export_dir = "storage/saved_models/exports/"
        self.agent_id = round(time.time())
        self.agent_directory = f"{BASE_SAVE_PATH}/{self.agent_id}/"
        if _make_dirs:
            os.makedirs(self.model_export_dir, exist_ok=True)
            os.makedirs(self.agent_directory)

        if os.path.isdir("storage/experience"):
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
        self.policy_loss_history = []
        self.value_loss_history = []
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
        # TODO need to deal with the fact that old_action_prob fucks me
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
            return tf.reduce_mean(gaussian_entropy(stdevs=tf.split(policy_output, 2, axis=-1)))
        else:
            return tf.reduce_mean(categorical_entropy(policy_output))

    def drill(self, n: int, epochs: int, batch_size: int, monitor=None, export: bool = False, save_every: int = 0,
              separate_eval: bool = False) -> "PPOAgent":
        """Start a training loop of the agent.
        
        Runs **n** cycles of experience gathering and optimization based on the gathered experience.

        Args:
            n (int): the number of experience-optimization cycles that shall be run
            epochs (int): the number of epochs for which the model is optimized on the same experience data
            batch_size (int): batch size for the optimization
            monitor: story telling object that creates visualizations of the training process on the fly (Default
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
        if self.is_recurrent and batch_size > self.workers:
            logging.warning(
                f"Batchsize is larger than possible with the available number of independent sequences for "
                f"Truncated BPTT. Setting batchsize to {self.workers}, which means {self.workers * self.tbptt_length} "
                f"transitions per batch.")
            batch_size = self.workers

        # rebuild model with desired batch size
        weights = self.joint.get_weights()
        self.policy, self.value, self.joint = self.model_builder(self.env, **({"bs": batch_size} if "bs" in fargs(
            self.model_builder).args else {}))
        self.joint.set_weights(weights)

        available_cpus = multiprocessing.cpu_count()
        ray.init(local_mode=self.debug, num_cpus=available_cpus, logging_level=logging.ERROR)
        print(f"Parallelizing {self.workers} Workers Over {available_cpus} Threads.\n")
        for self.iteration in range(self.iteration, n):
            time_dict = OrderedDict()
            subprocess_start = time.time()

            # run simulations in parallel
            flat_print("Gathering...")

            # export the current state of the policy and value network under unique (-enough) key
            name_key = round(time.time())
            if export:
                self.joint.save(f"{self.model_export_dir}/{name_key}/model")
                model_representation = f"{self.model_export_dir}/{name_key}/"
            else:
                model_representation = ModelTuple(self.builder_function_name, self.joint.get_weights())

            # create processes and execute them
            split_stats = ray.get([collect.remote(model_representation, self.horizon, self.env_name, self.discount,
                                                  self.lam, self.tbptt_length, pid) for pid in range(self.workers)])
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
                          "Consider activating separate evaluation in drill() to get meaningful statistics.")

                self.episode_length_history.extend(stats.episode_lengths)
                self.episode_reward_history.extend(stats.episode_rewards)
                self.cycle_length_history.append(None if stats.numb_completed_episodes == 0
                                                 else statistics.mean(stats.episode_lengths))
                self.cycle_reward_history.append(None if stats.numb_completed_episodes == 0
                                                 else statistics.mean(stats.episode_rewards))
            else:
                flat_print("Evaluating...")
                eval_lengths, eval_rewards = self.evaluate(8, ray_already_initialized=True)
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
            self.total_frames_seen += stats.numb_processed_frames
            self.total_episodes_seen += stats.numb_completed_episodes

            if monitor is not None and monitor.gif_every != 0 and (self.iteration + 1) % monitor.gif_every == 0:
                print("Creating Episode GIFs for current state of policy...")
                monitor.create_episode_gif(n=1)

            if monitor is not None and monitor.frequency != 0 and (self.iteration + 1) % monitor.frequency == 0:
                monitor.update()

            if save_every != 0 and self.iteration != 0 and (self.iteration + 1) % save_every == 0:
                self.save_agent_state()

            time_dict["finalizing"] = time.time() - subprocess_start

            # calculate processing speed in fps
            self.current_fps = stats.numb_processed_frames / (sum([v for v in time_dict.values() if v is not None]))
            self.time_dicts.append(time_dict)

        return self

    @tf.function
    def _learn_on_batch(self, batch):
        print("new_batch")
        # optimize policy and value network simultaneously
        with tf.GradientTape() as tape:
            state_batch = batch["state"] if "state" in batch else (batch["in_vision"], batch["in_proprio"],
                                                                   batch["in_touch"], batch["in_goal"])
            policy_output, value_output = self.joint(state_batch, training=True)
            old_values = batch["return"] - batch["advantage"]

            if self.continuous_control:
                # if action space is continuous, calculate PDF at chosen action value
                means, stdevs = tf.split(policy_output, 2, axis=-1)
                action_probabilities = gaussian_pdf(batch["action"],
                                                    means=means,
                                                    stdevs=stdevs)
            else:
                # if the action space is discrete, extract the probabilities of actions actually chosen
                action_probabilities = extract_discrete_action_probabilities(policy_output, batch["action"])

            # calculate the clipped loss
            policy_loss = self.policy_loss(action_prob=action_probabilities,
                                           old_action_prob=batch["action_prob"],
                                           advantage=batch["advantage"])
            value_loss = self.value_loss(value_predictions=tf.squeeze(value_output),
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

        return tf.reduce_mean(entropy), tf.reduce_mean(policy_loss), tf.reduce_mean(value_loss)

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
        progressbar = tqdm(total=epochs * ((self.horizon * self.workers / self.tbptt_length) / batch_size),
                           leave=False, desc="Optimizing")
        policy_loss_history, value_loss_history, entropy_history = [], [], []
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break correlation, then divided into batches
            # dataset = dataset.shuffle(10000)
            batched_dataset = dataset.batch(batch_size, drop_remainder=True)
            actor_epoch_losses, value_epoch_losses, entropies = [], [], []
            for b in batched_dataset:
                # use the dataset to optimize the model
                with tf.device(self.device):
                    if not self.is_recurrent:
                        ent, pi_loss, v_loss = self._learn_on_batch(b)
                        progressbar.update(1)
                    else:
                        # truncated back propagation through time
                        # batch shape: (BATCH_SIZE, N_SUBSEQUENCES, SUBSEQUENCE_LENGTH, *STATE_DIMS)
                        split_batch = {k: tf.split(v, v.shape[1], axis=1) for k, v in b.items()}
                        for i in range(len(b["advantage"])):
                            partial_batch = {k: tf.squeeze(v[i]) for k, v in split_batch.items()}
                            ent, pi_loss, v_loss = self._learn_on_batch(partial_batch)
                            progressbar.update(1)

                entropies.append(ent)
                actor_epoch_losses.append(pi_loss)
                value_epoch_losses.append(v_loss)

            # reset RNN states after an epoch if there are any
            self.joint.reset_states()

            # remember some statistics
            policy_loss_history.append(tf.reduce_mean(actor_epoch_losses).numpy().item())
            value_loss_history.append(tf.reduce_mean(value_epoch_losses).numpy().item())
            entropy_history.append(tf.reduce_mean(entropies).numpy().item())

        # store statistics in agent history
        self.policy_loss_history.extend(policy_loss_history)
        self.value_loss_history.extend(value_loss_history)
        self.entropy_history.extend(entropy_history)

        progressbar.close()

    def evaluate(self, n: int, ray_already_initialized: bool = False) -> Tuple[List[int], List[int]]:
        """Evaluate the current state of the policy on the given environment for n episodes. Optionally can render to
        visually inspect the performance.

        Args:
            n (int): integer value indicating the number of episodes that shall be run
            ray_already_initialized (bool): if True, do not initialize ray again (default False)

        Returns:
            two lists of length n, giving episode lengths and rewards respectively
        """
        if not ray_already_initialized:
            ray.init(local_mode=self.debug, logging_level=logging.ERROR)

        model_representation = ModelTuple(self.builder_function_name, self.policy.get_weights())
        result_ids = [evaluate.remote(model_representation, self.env_name) for _ in range(n)]

        lengths, rewards = zip(*[ray.get(oi) for oi in result_ids])

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
        if isinstance(self.lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = self.lr_schedule(self.optimizer.iterations)
        else:
            current_lr = self.lr_schedule
        flat_print(f"{sc}{f'Iteration {self.iteration:5d}' if self.iteration != 0 else 'Before Training'}{ec}: "
                   f"AvgRew.: {nc}{0 if self.cycle_reward_history[-1] is None else round(self.cycle_reward_history[-1], 2):8.2f}{ec}; "
                   f"AvgLen.: {nc}{0 if self.cycle_length_history[-1] is None else round(self.cycle_length_history[-1], 2):8.2f}{ec}; "
                   f"AvgEnt.: {nc}{0 if len(self.entropy_history) == 0 else round(self.entropy_history[-1], 2):5.2f}{ec}; "
                   f"Eps.: {nc}{self.total_episodes_seen:5d}{ec}; "
                   f"Lr: {nc}{current_lr:.2e}{ec}; "
                   f"Updates: {nc}{self.optimizer.iterations.numpy().item():6d}{ec}; "
                   f"Frames: {nc}{round(self.total_frames_seen / 1e3, 3):8.3f}{ec}k; "
                   f"Speed: {nc}{self.current_fps:7.2f}{ec}fps {time_distribution_string}\n")

    def save_agent_state(self):
        """Save the current state of the agent into the agent directory, identified by the current iteration."""
        self.joint.save_weights(self.agent_directory + f"/{self.iteration}/weights")

        with open(self.agent_directory + f"/{self.iteration}/parameters.json", "w") as f:
            json.dump(self.get_parameters(), f)

    def get_parameters(self):
        """Get the agents parameters necessary to reconstruct it."""
        parameters = self.__dict__.copy()
        del parameters["env"]
        del parameters["policy"], parameters["value"], parameters["joint"]
        del parameters["optimizer"], parameters["lr_schedule"], parameters["model_builder"]

        parameters["c_entropy"] = parameters["c_entropy"].numpy().item()
        parameters["c_value"] = parameters["c_value"].numpy().item()

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
        print(f"Loading from most recent iteration {latest}.")
        with open(f"{agent_path}/{latest}/parameters.json", "r") as f:
            parameters = json.load(f)

        model_builder = getattr(models, parameters["builder_function_name"])

        env = gym.make(parameters["env_name"])

        loaded_agent = PPOAgent(model_builder,
                                environment=env,
                                horizon=parameters["horizon"],
                                workers=parameters["workers"],
                                learning_rate=parameters["learning_rate"],
                                discount=parameters["discount"],
                                lam=parameters["lam"],
                                clip=parameters["clip"],
                                c_entropy=parameters["c_entropy"],
                                c_value=parameters["c_value"],
                                gradient_clipping=parameters["gradient_clipping"],
                                clip_values=parameters["clip_values"],
                                tbptt_length=parameters["tbptt_length"],
                                _make_dirs=False)

        for p, v in parameters.items():
            loaded_agent.__dict__[p] = v

        loaded_agent.joint.load_weights(f"{BASE_SAVE_PATH}/{agent_id}/" + f"/{latest}/weights")

        return loaded_agent
