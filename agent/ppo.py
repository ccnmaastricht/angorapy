#!/usr/bin/env python
"""Implementation of Proximal Policy Optimization Algorithm."""
import json
import logging
import os
import re
import statistics
import time
from collections import OrderedDict
from glob import glob
from typing import Union, Tuple, Any

import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box, Dict
from mpi4py import MPI
from tensorflow.keras.optimizers import Optimizer
from tqdm import tqdm

import models
from agent import policies

from agent.core import extract_discrete_action_probabilities
from agent.dataio import read_dataset_from_storage
from agent.gather import Gatherer

from agent.policies import BasePolicyDistribution, CategoricalPolicyDistribution, GaussianPolicyDistribution
from utilities import const

from utilities.const import COLORS, BASE_SAVE_PATH, PRETRAINED_COMPONENTS_PATH, STORAGE_DIR
from utilities.const import MIN_STAT_EPS
from utilities.datatypes import mpi_condense_stats, StatBundle
from utilities.model_utils import is_recurrent_model, get_layer_names, get_component, reset_states_masked, \
    requires_batch_size
from utilities.statistics import ignore_none
from utilities.util import mpi_flat_print, env_extract_dims, add_state_dims, merge_into_batch, detect_finished_episodes, \
    make_env
from utilities.wrappers import CombiWrapper, SkipWrapper, BaseRunningMeanWrapper, mpi_merge_wrappers


HOROVOD = False
INIT_HOROVOD = False

# get COMM and find gpus
mpi_comm = MPI.COMM_WORLD
gpus = tf.config.list_physical_devices('GPU')
is_gpu_process = MPI.COMM_WORLD.rank < len(gpus)

if not is_gpu_process:
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # pass
    tf.config.experimental.set_memory_growth(gpus[mpi_comm.rank], True)

if INIT_HOROVOD:
    import horovod.tensorflow as hvd

    # create subcomm with GPUs
    gpu_subcomm = mpi_comm.Split(color=int(is_gpu_process))
    hvd.init(comm=gpu_subcomm)

    # prevent full blockage of VRAM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if is_gpu_process:
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[mpi_comm.rank], 'GPU')


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
                 discount: float = 0.99, lam: float = 0.95, clip: float = 0.2, c_entropy: float = 0.01,
                 c_value: float = 0.5, gradient_clipping: float = None, clip_values: bool = True,
                 tbptt_length: int = 16, lr_schedule: str = None, distribution: BasePolicyDistribution = None,
                 reward_configuration: str = None, preprocessor=None, _make_dirs=True, debug: bool = False,
                 pretrained_components: list = None):
        """ Initialize the PPOAgent with given hyperparameters. Policy and value network will be freshly initialized.

        Args:
            preprocessor:
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
        self.preprocessor = preprocessor if preprocessor is not None else SkipWrapper()
        self.preprocessor.warmup(self.env)

        if MPI.COMM_WORLD.rank == 0:
            print(f"Using {self.preprocessor} for preprocessing.")

        # hyperparameters
        self.horizon = horizon
        self.n_workers = workers
        self.discount = discount
        self.learning_rate = learning_rate
        self.clip = clip
        self.c_entropy = tf.constant(c_entropy, dtype=tf.float32)
        self.c_value = tf.constant(c_value, dtype=tf.float32)
        self.lam = lam
        self.gradient_clipping = gradient_clipping if gradient_clipping != 0 else None
        self.clip_values = clip_values
        self.tbptt_length = tbptt_length
        self.reward_configuration = reward_configuration

        # learning rate schedule
        self.lr_schedule_type = lr_schedule
        if lr_schedule is None:
            # adjust learning rate based on number of parallel GPUs
            self.lr_schedule = self.learning_rate * (hvd.size() if HOROVOD else 1)
        elif lr_schedule.lower() == "exponential":
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate * (hvd.size() if HOROVOD else 1),
                decay_steps=workers * horizon,  # decay after every cycle
                decay_rate=0.98
            )
        else:
            raise ValueError("Unknown Schedule type. Choose one of (None, exponential)")

        # models and optimizers
        self.distribution = distribution
        if self.distribution is None:
            self.distribution = CategoricalPolicyDistribution(
                self.env) if not self.continuous_control else GaussianPolicyDistribution(self.env)
        assert self.continuous_control == self.distribution.is_continuous, "Invalid distribution for environment."
        self.model_builder = model_builder
        self.builder_function_name = model_builder.__name__
        self.policy, self.value, self.joint = model_builder(self.env, self.distribution,
                                                            **({"bs": 1} if requires_batch_size(model_builder) else {}))

        if pretrained_components is not None:
            print("Loading pretrained components:")
            for pretraining_name in pretrained_components:
                component_path = os.path.join(PRETRAINED_COMPONENTS_PATH, f"{pretraining_name}.h5")
                if os.path.isfile(component_path):
                    component = tf.keras.models.load_model(component_path, compile=False)
                else:
                    print(f"\tNo such pretraining found at {component_path}. Skipping.")
                    continue

                if component.name in get_layer_names(self.joint):
                    try:
                        get_component(self.joint, component.name).set_weights(component.get_weights())
                        print(f"\tSuccessfully loaded component '{component.name}'")
                    except ValueError as e:
                        print(f"Could not load weights into component: {e}")
                else:
                    print(f"\tNo outer component named '{component.name}' in model. Skipping.")

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
        self.gathering_fps = 0
        self.optimization_fps = 0
        self.device = "CPU:0"
        self.model_export_dir = "storage/saved_models/exports/"
        self.agent_id = mpi_comm.bcast(round(time.time()), root=0)
        self.agent_directory = f"{BASE_SAVE_PATH}/{self.agent_id}/"
        if _make_dirs:
            os.makedirs(self.model_export_dir, exist_ok=True)
            os.makedirs(self.agent_directory, exist_ok=True)

        if os.path.isdir("storage/experience"):
            # TODO loading a currently training agent will potentially delete its storage
            # shutil.rmtree("storage/experience/")
            pass
        os.makedirs("storage/experience/", exist_ok=True)

        # statistics
        self.total_frames_seen = 0
        self.total_episodes_seen = 0
        self.episode_reward_history = []
        self.episode_length_history = []
        self.cycle_reward_history = []
        self.cycle_length_history = []
        self.cycle_reward_std_history = []
        self.cycle_length_std_history = []
        self.cycle_stat_n_history = []
        self.entropy_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.time_dicts = []
        self.cycle_timings = []
        self.underflow_history = []

        self.preprocessor_stat_history = {
            w.__class__.__name__: {"mean": [w.simplified_mean()], "stdev": [w.simplified_stdev()]}
            for w in self.preprocessor if isinstance(w, BaseRunningMeanWrapper)
        }

    def __repr__(self):
        return f"PPOAgent[at {self.iteration}][{self.env_name}]"

    def set_gpu(self, activated: bool):
        """Set GPU usage mode."""
        self.device = "GPU:0" if activated else "CPU:0"

    def policy_loss(self, action_prob: tf.Tensor, old_action_prob: tf.Tensor, advantage: tf.Tensor) -> tf.Tensor:
        """Actor's clipped objective as given in the PPO paper. Original objective is to be maximized
        (as given in the paper), but this is the negated objective to be minimized! In the recurrent version
        a mask is calculated based on 0 values in the old_action_prob tensor. This mask is then applied in the mean
        operation of the loss.

        Args:
          action_prob (tf.Tensor): the probability of the action for the state under the current policy
          old_action_prob (tf.Tensor): the probability of the action taken given by the old policy during the episode
          advantage (tf.Tensor): the advantage that taking the action gives over the estimated state value

        Returns:
          the value of the objective function

        """
        r = tf.exp(action_prob - old_action_prob)
        clipped = tf.maximum(
            tf.math.multiply(r, -advantage),
            tf.math.multiply(tf.clip_by_value(r, 1 - self.clip, 1 + self.clip), -advantage)
        )

        if self.is_recurrent:
            # build and apply a mask over the probabilities (recurrent)
            mask = tf.not_equal(old_action_prob, 0)
            clipped_masked = tf.where(mask, clipped, 0)
            return tf.reduce_sum(clipped_masked) / tf.reduce_sum(tf.cast(mask, tf.float32))
        else:
            return tf.reduce_mean(clipped)

    def value_loss(self, value_predictions: tf.Tensor, old_values: tf.Tensor, returns: tf.Tensor,
                   old_action_prob: tf.Tensor, clip: bool = True) -> tf.Tensor:
        """Loss of the critic network as squared error between the prediction and the sampled future return. In the
        recurrent case a mask is calculated based on 0 values in the old_action_prob tensor. This mask is then applied
        in the mean operation of the loss.

        Args:
          value_predictions (tf.Tensor): value prediction by the current critic network
          old_values (tf.Tensor): value prediction by the old critic network during gathering
          returns (tf.Tensor): discounted return estimation
          old_action_prob (tf.Tensor): probabilities from old policy, used to determine mask
          clip (object): (Default value = True) value loss can be clipped by same range as policy loss

        Returns:
          squared error between prediction and return
        """
        error = tf.square(value_predictions - returns)

        if clip:
            # clips value error to reduce variance
            clipped_values = old_values + tf.clip_by_value(value_predictions - old_values, -self.clip, self.clip)
            clipped_error = tf.square(clipped_values - returns)
            error = tf.maximum(clipped_error, error)

        if self.is_recurrent:
            # build and apply a mask over the old values (recurrent)
            mask = tf.not_equal(old_action_prob, 0)
            error_masked = tf.where(mask, error, 0)  # masking with tf.where because inf * 0 = nan...
            return (tf.reduce_sum(error_masked) / tf.reduce_sum(tf.cast(mask, tf.float32))) * 0.5
        else:
            return tf.reduce_mean(error) * 0.5

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
        return tf.reduce_mean(self.distribution.entropy(policy_output))

    def drill(self, n: int, epochs: int, batch_size: int, monitor=None, save_every: int = 0,
              separate_eval: bool = False, stop_early: bool = True, radical_evaluation=False) -> "PPOAgent":
        """Start a training loop of the agent.
        
        Runs **n** cycles of experience gathering and optimization based on the gathered experience.

        Args:
            n (int): the number of experience-optimization cycles that shall be run
            epochs (int): the number of epochs for which the model is optimized on the same experience data
            batch_size (int): batch size for the optimization
            monitor: story telling object that creates visualizations of the training process on the fly (Default
                value = None)
            save_every (int): for any int x > 0 save the policy every x iterations, if x = 0 (default) do not save
            separate_eval (bool): if false (default), use episodes from gathering for statistics, if true, evaluate 10
                additional episodes.
            stop_early (bool): if true, stop the drill early if at least the previous 5 cycles achieved a performance
                above the environments threshold


        Returns:
            self
        """
        # determine the number of independent repeated gatherings required on this worker
        base, extra = divmod(self.n_workers, MPI.COMM_WORLD.size)
        worker_split = [base + (r < extra) for r in range(MPI.COMM_WORLD.size)]
        worker_collection_ids = list(range(self.n_workers))[
                                sum(worker_split[:mpi_comm.rank]):sum(worker_split[:mpi_comm.rank + 1])]
        list_of_collection_id_lists = mpi_comm.gather(worker_collection_ids, root=0)

        if MPI.COMM_WORLD.rank == 0:
            print(
                f"\n\nDrill started using {MPI.COMM_WORLD.size} processes of which {len(gpus)} are GPU optimizers."
                f" Worker distribution: {[base + (r < extra) for r in range(MPI.COMM_WORLD.size)]}.\n"
                f"IDs over Workers: {list_of_collection_id_lists}")

            assert self.horizon * self.n_workers >= batch_size, "Batch Size is larger than the number of transitions."

        assert self.horizon * self.n_workers >= batch_size, "Batch Size is larger than the number of transitions."
        if mpi_comm.rank == 0 and self.is_recurrent and batch_size > self.n_workers:
            logging.warning(
                f"Batchsize is larger than possible with the available number of independent sequences for "
                f"Truncated BPTT. Setting batchsize to {self.n_workers}, which means {self.n_workers * self.tbptt_length} "
                f"transitions per batch.")
            batch_size = self.n_workers

            if mpi_comm.rank == 0:
                logging.warning(
                    f"Batchsize is larger than possible with the available number of independent sequences for "
                    f"Truncated BPTT. Setting batchsize to {self.n_workers}, which means {self.n_workers * self.tbptt_length} "
                    f"transitions per batch.")

        # rebuild model with desired batch size
        weights = self.joint.get_weights()
        self.policy, self.value, self.joint = self.model_builder(self.env, self.distribution, **(
            {"bs": batch_size} if requires_batch_size(self.model_builder) else {}))
        self.joint.set_weights(weights)

        actor = self._make_actor()

        cycle_start = None
        full_drill_start_time = time.time()
        for self.iteration in range(self.iteration, n):
            mpi_flat_print(f"Gathering cycle {self.iteration}...")

            time_dict = OrderedDict()
            subprocess_start = time.time()

            # distribute parameters from rank 0 to all other ranks
            values = mpi_comm.bcast(self.joint.get_weights(), root=0)
            self.joint.set_weights(values)
            actor.update_weights(self.joint.get_weights())

            # run simulations in parallel
            worker_stats, worker_preprocessors = [], []
            serialized_wrapper = self.preprocessor.serialize()
            for i in worker_collection_ids:
                collection = actor.collect(self.horizon, self.discount, self.lam, self.tbptt_length, serialized_wrapper,
                                           collector_id=i)
                worker_stats.append(collection[0])
                worker_preprocessors.append(collection[1])

            # merge gatherings from all workers
            stats = mpi_condense_stats(worker_stats)
            self.preprocessor = mpi_merge_wrappers(worker_preprocessors, self.preprocessor)
            self.preprocessor = mpi_comm.bcast(self.preprocessor, root=0)

            time_dict["gathering"] = time.time() - subprocess_start
            subprocess_start = time.time()

            # make seperate evaluation if necessary and wanted
            stats = mpi_comm.bcast(stats, root=0)
            stats_with_evaluation = stats
            if separate_eval:
                if radical_evaluation or stats.numb_completed_episodes < MIN_STAT_EPS:
                    mpi_flat_print("Evaluating...")
                    n_evaluations = MIN_STAT_EPS if radical_evaluation else MIN_STAT_EPS - stats.numb_completed_episodes
                    evaluation_stats, _ = self.evaluate(n_evaluations, actor=actor)

                    if radical_evaluation:
                        stats_with_evaluation = evaluation_stats
                    else:
                        stats_with_evaluation = mpi_condense_stats([stats, evaluation_stats])
            elif MPI.COMM_WORLD.rank == 0 and stats.numb_completed_episodes == 0:
                print("WARNING: You are using a horizon that caused this cycle to not finish a single episode. "
                      "Consider activating separate evaluation in drill() to get meaningful statistics.")

            if not is_gpu_process:
                # only the GPU processes optimize and calculate the rest
                continue

            if mpi_comm.rank == 0:
                # record stats and preprocessor in the agent
                self.record_preprocessor(self.preprocessor)
                self.record_stats(stats_with_evaluation)

                time_dict["evaluating"] = time.time() - subprocess_start
                subprocess_start = time.time()

                # save if best
                if self.cycle_reward_history[-1] is not None and self.cycle_reward_history[-1] == ignore_none(max,
                                                                                                              self.cycle_reward_history):
                    self.save_agent_state(name="best")

            # early stopping  TODO check how this goes with MPI
            if self.env.spec.reward_threshold is not None and stop_early:
                if np.all(np.greater_equal(self.cycle_reward_history[-5:], self.env.spec.reward_threshold)):
                    print("\rAll catch a breath, we stop the drill early due to the formidable result!")
                    break

            if mpi_comm.rank == 0:
                if cycle_start is not None:
                    self.cycle_timings.append(round(time.time() - cycle_start, 2))
                self.report(total_iterations=n)
                cycle_start = time.time()

            # PARALLELIZED OPTIMIZATION
            mpi_flat_print("Optimizing...")
            dataset = read_dataset_from_storage(dtype_actions=tf.float32 if self.continuous_control else tf.int32,
                                                is_shadow_hand=isinstance(self.state_dim, tuple),
                                                id_prefix=self.agent_id)
            self.optimize(dataset, epochs, batch_size)

            if mpi_comm.rank == 0:
                time_dict["optimizing"] = time.time() - subprocess_start

                mpi_flat_print("Finalizing...")
                self.total_frames_seen += stats.numb_processed_frames
                self.total_episodes_seen += stats.numb_completed_episodes

                # update monitor logs
                if monitor is not None:
                    if monitor.gif_every != 0 and (self.iteration + 1) % monitor.gif_every == 0:
                        print("Creating Episode GIFs for current state of policy...")
                        monitor.create_episode_gif(n=1)

                    if monitor.frequency != 0 and (self.iteration + 1) % monitor.frequency == 0:
                        monitor.update()

                # save the current state of the agent
                if save_every != 0 and self.iteration != 0 and (self.iteration + 1) % save_every == 0:
                    self.save_agent_state()

                # calculate processing speed in fps
                self.current_fps = stats.numb_processed_frames / (sum([v for v in time_dict.values() if v is not None]))
                self.gathering_fps = (stats.numb_processed_frames // min(self.n_workers, MPI.COMM_WORLD.size)) / (
                    time_dict["gathering"])
                self.optimization_fps = (stats.numb_processed_frames * epochs) / (time_dict["optimizing"])
                self.time_dicts.append(time_dict)

        if mpi_comm.rank == 0:
            print(f"Drill finished after {round(time.time() - full_drill_start_time, 2)}s.")

        return self

    def record_stats(self, stats):
        """Record a given StatsBundle in the history of the agent."""
        try:
            mean_eps_length = statistics.mean(stats.episode_lengths) if len(
                stats.episode_lengths) > 1 else stats.episode_lengths[0]
            mean_eps_rewards = statistics.mean(stats.episode_rewards) if len(
                stats.episode_rewards) > 1 else stats.episode_rewards[0]
        except IndexError:
            mean_eps_rewards = None
            mean_eps_length = None

        stdev_eps_length = statistics.stdev(stats.episode_lengths) if len(stats.episode_lengths) > 1 else 0
        stdev_eps_rewards = statistics.stdev(stats.episode_rewards) if len(stats.episode_rewards) > 1 else 0

        self.episode_length_history.append(stats.episode_lengths)
        self.episode_reward_history.append(stats.episode_rewards)
        self.cycle_length_history.append(None if stats.numb_completed_episodes == 0 else mean_eps_length)
        self.cycle_reward_history.append(None if stats.numb_completed_episodes == 0 else mean_eps_rewards)
        self.cycle_length_std_history.append(None if stats.numb_completed_episodes <= 1 else stdev_eps_length)
        self.cycle_reward_std_history.append(None if stats.numb_completed_episodes <= 1 else stdev_eps_rewards)
        self.cycle_stat_n_history.append(stats.numb_completed_episodes)
        self.underflow_history.append(stats.tbptt_underflow)

    def record_preprocessor(self, preprocessor):
        """Record the stats of a given preprocessor in the history of the agent."""
        for w in preprocessor:
            if w.name not in self.preprocessor_stat_history.keys() or not isinstance(w, BaseRunningMeanWrapper):
                continue

            self.preprocessor_stat_history[w.name]["mean"].append(w.simplified_mean())
            self.preprocessor_stat_history[w.name]["stdev"].append(w.simplified_stdev())

    def _make_actor(self) -> Gatherer:
        actor = Gatherer(self.builder_function_name,
                         self.distribution.__class__.__name__,
                         self.env_name,
                         MPI.COMM_WORLD.rank,
                         self.agent_id,
                         reward_configuration=self.reward_configuration)

        return actor

    @tf.function
    def _learn_on_batch(self, batch):
        # optimize policy and value network simultaneously
        with tf.GradientTape() as tape:
            state_batch = batch["state"] if "state" in batch else (batch["in_vision"], batch["in_proprio"],
                                                                   batch["in_touch"], batch["in_goal"])
            policy_output, value_output = self.joint(state_batch, training=True)
            old_values = batch["value"]

            if self.continuous_control:
                # if action space is continuous, calculate PDF at chosen action value
                action_probabilities = self.distribution.log_probability(batch["action"], *policy_output)
            else:
                # if the action space is discrete, extract the probabilities of actions actually chosen
                action_probabilities = extract_discrete_action_probabilities(policy_output, batch["action"])

            # calculate the clipped loss
            policy_loss = self.policy_loss(action_prob=action_probabilities, old_action_prob=batch["action_prob"],
                                           advantage=batch["advantage"])
            value_loss = self.value_loss(value_predictions=tf.squeeze(value_output, axis=-1), old_values=old_values,
                                         returns=batch["return"], old_action_prob=batch["action_prob"],
                                         clip=self.clip_values)
            entropy = self.entropy_bonus(policy_output)
            total_loss = policy_loss + tf.multiply(self.c_value, value_loss) - tf.multiply(self.c_entropy, entropy)

        # wrap for distribution
        if HOROVOD:
            tape = hvd.DistributedGradientTape(tape)

        # calculate the gradient of the joint model based on total loss
        gradients = tape.gradient(total_loss, self.joint.trainable_variables)

        # clip gradients to avoid gradient explosion and stabilize learning
        if self.gradient_clipping is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)

        # apply the gradients to the joint model's parameters
        self.optimizer.apply_gradients(zip(gradients, self.joint.trainable_variables))

        info = {
            "policy_output": policy_output,
            # "actions": batch["action"],
            # "action_probabilities": action_probabilities,
            # "old_action_probabilities": batch["action_prob"],
            # "value_output": value_output,
            # "gradients": gradients
        }

        return tf.reduce_mean(entropy), tf.reduce_mean(policy_loss), tf.reduce_mean(value_loss), info

    def optimize(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> None:
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
        progressbar = tqdm(total=epochs * ((self.horizon * self.n_workers / self.tbptt_length) / batch_size),
                           leave=False, desc="Optimizing", disable=True)
        policy_loss_history, value_loss_history, entropy_history = [], [], []
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break correlation
            if not self.is_recurrent:
                dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)

            if HOROVOD:
                # dataset = dataset.take(dataset.cardinality() // hvd.size())  # todo wont work before 2.3
                dataset = dataset

            # then divide into batches
            batched_dataset = dataset.batch(batch_size, drop_remainder=True)

            first_batch = True
            policy_epoch_losses, value_epoch_losses, entropies = [], [], []
            for b in batched_dataset:
                # use the dataset to optimize the model
                with tf.device(self.device):
                    if not self.is_recurrent:
                        ent, pi_loss, v_loss, info = self._learn_on_batch(b)
                        progressbar.update(1)
                    else:
                        # truncated back propagation through time
                        # batch shape: (BATCH_SIZE, N_SUBSEQUENCES, SUBSEQUENCE_LENGTH, *STATE_DIMS)
                        split_batch = {k: tf.split(v, v.shape[1], axis=1) for k, v in b.items()}
                        for i in range(len(split_batch["advantage"])):
                            # extract subsequence and squeeze away the N_SUBSEQUENCES dimension
                            partial_batch = {k: tf.squeeze(v[i], axis=1) for k, v in split_batch.items()}
                            ent, pi_loss, v_loss, info = self._learn_on_batch(partial_batch)

                            progressbar.update(1)

                            # make partial RNN state resets
                            reset_mask = detect_finished_episodes(partial_batch["action_prob"])
                            reset_states_masked(self.joint, reset_mask)

                if first_batch and HOROVOD:
                    hvd.broadcast_variables(self.joint.variables, root_rank=0)
                    hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
                first_batch = False

                entropies.append(ent)
                policy_epoch_losses.append(pi_loss)
                value_epoch_losses.append(v_loss)

                # reset RNN states after each outer batch
                self.joint.reset_states()

            # remember some statistics
            policy_loss_history.append(tf.reduce_mean(policy_epoch_losses).numpy().item())
            value_loss_history.append(tf.reduce_mean(value_epoch_losses).numpy().item())
            entropy_history.append(tf.reduce_mean(entropies).numpy().item())

        # store statistics in agent history
        self.policy_loss_history.append(statistics.mean(policy_loss_history))
        self.value_loss_history.append(statistics.mean(value_loss_history))
        self.entropy_history.append(statistics.mean(entropy_history))

        progressbar.close()

    def evaluate(self, n: int, actor: Gatherer = None, save: bool = False) -> Tuple[StatBundle, Any]:
        """Evaluate the current state of the policy on the given environment for n episodes. Optionally can render to
        visually inspect the performance.

        Args:
            n (int): integer value indicating the number of episodes that shall be run
            actor (Gatherer): actor object to be used for evaluation
            save (bool): whether to save the evaluation to the monitor directory

        Returns:
            StatBundle with evaluation results
        """
        actor = actor if actor is not None else self._make_actor()

        values = mpi_comm.bcast(self.joint.get_weights(), root=0)
        self.joint.set_weights(values)

        evaluation_result = actor.evaluate(self.preprocessor.serialize())
        gathered_evaluation_result = mpi_comm.gather(evaluation_result, root=0)

        stats, classes = None, None
        if MPI.COMM_WORLD.rank == 0:
            lengths, rewards, classes = zip(*gathered_evaluation_result)
            stats = StatBundle(n, sum(lengths), rewards, lengths, tbptt_underflow=0)

        stats = mpi_comm.bcast(stats, root=0)
        classes = mpi_comm.bcast(classes, root=0)

        if save and MPI.COMM_WORLD.rank == 0:
            os.makedirs(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/", exist_ok=True)

            previous_evaluations = {}
            if os.path.isfile(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/evaluations.json"):
                with open(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/evaluations.json", "r") as jf:
                    previous_evaluations = json.load(jf)

            previous_evaluations.update({self.iteration: stats._asdict()})

            with open(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/evaluations.json", "w") as jf:
                json.dump(previous_evaluations, jf)

        return stats, classes

    def report(self, total_iterations):
        """Print a report of the current state of the training."""
        if MPI.COMM_WORLD.rank != 0:
            return

        sc, nc, ec, ac = COLORS["OKGREEN"], COLORS["OKBLUE"], COLORS["ENDC"], COLORS["FAIL"]
        reward_col = ac
        if hasattr(self.env.spec, "reward_threshold") and self.env.spec.reward_threshold is not None and \
                self.cycle_reward_history[0] is not None and self.cycle_reward_history[-1] is not None:
            half_way_there_threshold = (self.cycle_reward_history[0]
                                        + 0.5 * (self.env.spec.reward_threshold - self.cycle_reward_history[0]))
            if self.env.spec.reward_threshold < self.cycle_reward_history[-1]:
                reward_col = COLORS["GREEN"]
            elif self.cycle_reward_history[-1] > half_way_there_threshold:
                reward_col = COLORS["ORANGE"]

        # calculate percentages of computation spend on different phases of the iteration
        time_distribution_string = ""
        if len(self.time_dicts) > 0:
            times = [time for time in self.time_dicts[-1].values() if time is not None]
            time_percentages = [str(round(100 * t / sum(times))) for i, t in enumerate(times)]
            time_distribution_string = "[" + "|".join(map(str, time_percentages)) + "]"
        if isinstance(self.lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = self.lr_schedule(self.optimizer.iterations)
        else:
            current_lr = self.lr_schedule

        # make fps string
        fps_string = f"[{nc}{int(round(self.gathering_fps)):6d}{ec}|{nc}{int(round(self.optimization_fps)):7d}{ec}]"

        # losses
        pi_loss = "-" if len(self.policy_loss_history) == 0 else f"{round(self.policy_loss_history[-1], 2):6.2f}"
        v_loss = "-" if len(self.value_loss_history) == 0 else f"{round(self.value_loss_history[-1], 2):8.2f}"
        ent = "-" if len(self.entropy_history) == 0 else f"{round(self.entropy_history[-1], 2):6.2f}"

        # tbptt underflow
        underflow = f"w: {nc}{self.underflow_history[-1]}{ec}; " if self.underflow_history[-1] is not None else ""

        # timings
        time_left = "unknown time"
        if len(self.cycle_timings) > 0:
            time_left = f"{round(ignore_none(statistics.mean, self.cycle_timings) * (total_iterations - self.iteration) / 60, 1)}mins"

        # print the report
        if mpi_comm.rank == 0:
            mpi_flat_print(
                f"{sc}{f'Cycle {self.iteration:5d}/{total_iterations}' if self.iteration != 0 else 'Before Training'}{ec}: "
                f"r: {reward_col}{'-' if self.cycle_reward_history[-1] is None else f'{round(self.cycle_reward_history[-1], 2):8.2f}'}{ec}; "
                f"len: {nc}{'-' if self.cycle_length_history[-1] is None else f'{round(self.cycle_length_history[-1], 2):8.2f}'}{ec}; "
                f"n: {nc}{'-' if self.cycle_stat_n_history[-1] is None else f'{self.cycle_stat_n_history[-1]:3d}'}{ec}; "
                f"loss: [{nc}{pi_loss}{ec}|{nc}{v_loss}{ec}|{nc}{ent}{ec}]; "
                f"eps: {nc}{self.total_episodes_seen:5d}{ec}; "
                f"lr: {nc}{current_lr:.2e}{ec}; "
                f"upd: {nc}{self.optimizer.iterations.numpy().item():6d}{ec}; "
                f"f: {nc}{round(self.total_frames_seen / 1e3, 3):8.3f}{ec}k; "
                f"{underflow}"
                f"fps: {fps_string} {time_distribution_string}; "
                f"took {self.cycle_timings[-1] if len(self.cycle_timings) > 0 else ''}s [{time_left} left]\n")

    def save_agent_state(self, name=None):
        """Save the current state of the agent into the agent directory, identified by the current iteration."""
        if name is None:
            name = str(self.iteration)

        self.joint.save_weights(os.path.join(self.agent_directory, f"{name}/weights"))
        with open(self.agent_directory + f"/{name}/parameters.json", "w") as f:
            json.dump(self.get_parameters(), f)

    def get_parameters(self):
        """Get the agents parameters necessary to reconstruct it."""
        parameters = self.__dict__.copy()
        del parameters["env"]
        del parameters["policy"], parameters["value"], parameters["joint"], parameters["distribution"]
        del parameters["optimizer"], parameters["lr_schedule"], parameters["model_builder"], parameters["preprocessor"]

        parameters["c_entropy"] = parameters["c_entropy"].numpy().item()
        parameters["c_value"] = parameters["c_value"].numpy().item()

        parameters["preprocessor"] = self.preprocessor.serialize()
        parameters["distribution"] = self.distribution.__class__.__name__

        return parameters

    @staticmethod
    def from_agent_state(agent_id: int, from_iteration: Union[int, str] = None, force_env_name=None,
                         path_modifier="") -> "PPOAgent":
        """Build an agent from a previously saved state.

        Args:
            agent_id:           the ID of the agent to be loaded
            from_iteration:     from which iteration to load, if None (default) use most recent, can be iteration int
                                or ["b", "best"] for best, if such was saved.

        Returns:
            loaded_agent: a PPOAgent object of the same state as the one saved into the path specified by agent_id
        """
        # TODO also load the state of the optimizers
        agent_path = path_modifier + BASE_SAVE_PATH + f"/{agent_id}"
        if not os.path.isdir(agent_path):
            raise FileNotFoundError(
                "The given agent ID does not match any existing save history from your current path.")

        if len(os.listdir(agent_path)) == 0:
            raise FileNotFoundError("The given agent ID's save history is empty.")

        latest_matches = PPOAgent.get_saved_iterations(agent_id)
        if from_iteration is None:
            if len(latest_matches) > 0:
                from_iteration = max(latest_matches)
            else:
                from_iteration = "best"

        if isinstance(from_iteration, str):
            assert from_iteration.lower() in ["best", "b"], "Unknown string identifier, can only be 'best'/'b' or int."
            from_iteration = "best"
        else:
            assert from_iteration in latest_matches, "There is no save at this iteration."

        print(f"Loading from iteration {from_iteration}.")
        with open(f"{agent_path}/{from_iteration}/parameters.json", "r") as f:
            parameters = json.load(f)

        env = make_env(parameters["env_name"] if force_env_name is None else force_env_name)
        model_builder = getattr(models, parameters["builder_function_name"])
        distribution = getattr(policies, parameters["distribution"])(env)
        preprocessor = CombiWrapper.from_serialization(parameters["preprocessor"])

        loaded_agent = PPOAgent(model_builder, environment=env, horizon=parameters["horizon"],
                                workers=parameters["n_workers"], learning_rate=parameters["learning_rate"],
                                discount=parameters["discount"], lam=parameters["lam"], clip=parameters["clip"],
                                c_entropy=parameters["c_entropy"], c_value=parameters["c_value"],
                                gradient_clipping=parameters["gradient_clipping"], preprocessor=preprocessor,
                                clip_values=parameters["clip_values"], tbptt_length=parameters["tbptt_length"],
                                lr_schedule=parameters["lr_schedule_type"], distribution=distribution, _make_dirs=False)

        for p, v in parameters.items():
            if p in ["distribution", "preprocessor"]:
                continue

            loaded_agent.__dict__[p] = v

        loaded_agent.joint.load_weights(f"{BASE_SAVE_PATH}/{agent_id}/" + f"/{from_iteration}/weights")

        return loaded_agent

    @staticmethod
    def get_saved_iterations(agent_id: int) -> list:
        """Return a list of iterations at which the agent of given ID has been saved."""
        agent_path = BASE_SAVE_PATH + f"/{agent_id}"

        if not os.path.isdir(agent_path):
            raise FileNotFoundError("The given agent ID does not match any existing save history.")

        if len(os.listdir(agent_path)) == 0:
            return []

        its = [int(i.group(0)) for i in [re.match("([0-9]+)", fn) for fn in os.listdir(agent_path)] if i is not None]

        return sorted(its)

    def finalize(self):
        """Perform final steps on the agent that are necessary no matter whether an error occurred or not."""
        print(f"{STORAGE_DIR}/{self.agent_id}_data_[0-9]+\.tfrecord")
        for file in glob(f"{STORAGE_DIR}/{self.agent_id}_data_[0-9]+\.tfrecord"):
            os.remove(file)
