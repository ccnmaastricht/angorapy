#!/usr/bin/env python
"""Implementation of Proximal Policy Optimization Algorithm."""
import gc
import json
import logging
import os
import re
import statistics
import time
from collections import OrderedDict
from glob import glob
from typing import Union, Tuple, Any, Callable

import gym
import numpy as np
import psutil
import tensorflow as tf
from gym.spaces import Discrete, Box
from mpi4py import MPI
from tensorflow.keras.optimizers import Optimizer

import models
from agent.dataio import read_dataset_from_storage
from agent.gather import Gatherer, evaluate
from agent.ppo.optim import learn_on_batch
from common import policies, const
from common.const import COLORS, BASE_SAVE_PATH, PRETRAINED_COMPONENTS_PATH, STORAGE_DIR
from common.const import MIN_STAT_EPS
from common.mpi_optim import MpiAdam
from common.policies import BasePolicyDistribution, CategoricalPolicyDistribution, GaussianPolicyDistribution
from common.transformers import BaseRunningMeanTransformer, transformers_from_serializations
from common.validators import validate_env_model_compatibility
from common.wrappers import BaseWrapper, make_env
from utilities.datatypes import mpi_condense_stats, StatBundle
from utilities.model_utils import is_recurrent_model, get_layer_names, get_component, reset_states_masked, \
    requires_batch_size, requires_sequence_length
from utilities.statistics import ignore_none
from utilities.util import mpi_flat_print, env_extract_dims, detect_finished_episodes

# get communicator and find optimization processes
mpi_comm = MPI.COMM_WORLD
gpus = tf.config.list_physical_devices('GPU')
is_root = mpi_comm.rank == 0

if len(gpus) > 0:
    node_names = list(set(mpi_comm.allgather(MPI.Get_processor_name())))
    node_comm = mpi_comm.Split(color=node_names.index(MPI.Get_processor_name()))
    node_optimizer_rank = node_comm.allreduce(mpi_comm.rank, op=MPI.MIN)

    # TODO make this work for multiple GPUs per node?
    is_optimization_process = MPI.COMM_WORLD.rank == node_optimizer_rank
else:
    is_optimization_process = True

if not is_optimization_process:
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

optimization_comm = mpi_comm.Split(color=(1 if is_optimization_process else 0))
n_optimizers = int(mpi_comm.allreduce(is_optimization_process, op=MPI.SUM))


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

    def __init__(self, model_builder: Callable, environment: BaseWrapper, horizon: int, workers: int,
                 learning_rate: float = 0.001, discount: float = 0.99, lam: float = 0.95, clip: float = 0.2,
                 c_entropy: float = 0.01, c_value: float = 0.5, gradient_clipping: float = None,
                 clip_values: bool = True, tbptt_length: int = 16, lr_schedule: str = None,
                 distribution: BasePolicyDistribution = None, reward_configuration: str = None, _make_dirs=True,
                 debug: bool = False, pretrained_components: list = None):
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
        self.env: BaseWrapper = environment
        self.env_name = self.env.unwrapped.spec.id
        self.state_dim, self.n_actions = env_extract_dims(self.env)
        if isinstance(self.env.action_space, Discrete):
            self.continuous_control = False
        elif isinstance(self.env.action_space, Box):
            self.continuous_control = True
        else:
            raise NotImplementedError(f"PPO cannot handle unknown Action Space Typ: {self.env.action_space}")

        self.env.warmup()

        if MPI.COMM_WORLD.rank == 0:
            print(f"Using {self.env.transformers} for preprocessing.")

        # hyperparameters
        self.horizon = horizon
        self.n_workers = workers
        self.discount = discount
        self.learning_rate = learning_rate
        self.clip = tf.constant(clip, dtype=tf.float32)
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
            self.lr_schedule = self.learning_rate
        elif lr_schedule.lower() == "exponential":
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=workers * horizon,  # decay after every cycle
                decay_rate=0.98)
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
        self.policy, self.value, self.joint = model_builder(
            env=self.env,
            distribution=self.distribution,
            **({"bs": 1} if requires_batch_size(model_builder) else {}),
            **({"sequence_length": self.tbptt_length} if requires_sequence_length(model_builder) else {}))
        validate_env_model_compatibility(self.env, self.joint)

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

        self.optimizer: Optimizer = MpiAdam(comm=optimization_comm, learning_rate=self.lr_schedule, epsilon=1e-5)
        self.is_recurrent = is_recurrent_model(self.policy)
        if not self.is_recurrent:
            self.tbptt_length = 1

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
        self.underflow_history = []

        # training statistics
        self.cycle_timings = []
        self.optimization_timings = []
        self.gathering_timings = []
        self.used_memory = []

        self.wrapper_stat_history = {}
        for transformer in self.env.transformers:
            self.wrapper_stat_history.update(
                {transformer.__class__.__name__: {"mean": [transformer.simplified_mean()],
                                                  "stdev": [transformer.simplified_stdev()]}})

    def record_wrapper_stats(self) -> None:
        """Records the stats from RunningMeanWrappers."""
        for transformer in self.env.transformers:
            if transformer.name not in self.wrapper_stat_history.keys() or not isinstance(transformer,
                                                                                          BaseRunningMeanTransformer):
                continue

            self.wrapper_stat_history[transformer.__class__.__name__]["mean"].append(transformer.simplified_mean())
            self.wrapper_stat_history[transformer.__class__.__name__]["stdev"].append(transformer.simplified_stdev())

    def __repr__(self):
        return f"PPOAgent[at {self.iteration}][{self.env_name}]"

    def set_gpu(self, activated: bool) -> None:
        """Set GPU usage mode."""
        self.device = "GPU:0" if activated else "CPU:0"
        pass

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
        worker_base, worker_extra = divmod(self.n_workers, MPI.COMM_WORLD.size)
        worker_split = [worker_base + (r < worker_extra) for r in range(MPI.COMM_WORLD.size)]
        worker_collection_ids = list(range(self.n_workers))[
                                sum(worker_split[:mpi_comm.rank]):sum(worker_split[:mpi_comm.rank + 1])]

        # determine the split of worker outputs over optimizers
        optimizer_base, optimizer_extra = divmod(self.n_workers, n_optimizers)
        optimizer_split = [optimizer_base + (r < optimizer_extra) for r in range(n_optimizers)]
        optimizer_collection_ids = list(range(self.n_workers))[
                                   sum(optimizer_split[:mpi_comm.rank]):sum(optimizer_split[:mpi_comm.rank + 1])]

        list_of_worker_collection_id_lists = mpi_comm.gather(worker_collection_ids, root=0)
        list_of_optimizer_collection_id_lists = mpi_comm.gather(optimizer_collection_ids, root=0)

        if is_root:
            print(
                f"\n\nDrill started using {MPI.COMM_WORLD.size} processes for {self.n_workers} workers of which "
                f"{n_optimizers} are optimizers."
                f" Worker distribution: {[worker_base + (r < worker_extra) for r in range(MPI.COMM_WORLD.size)]}.\n"
                f"IDs over Workers: {list_of_worker_collection_id_lists}\n"
                f"IDs over Optimizers: {list_of_optimizer_collection_id_lists}")

            assert self.horizon * self.n_workers >= batch_size, "Batch Size is larger than the number of transitions."
        assert self.horizon * self.n_workers >= batch_size, "Batch Size is larger than the number of transitions."

        if self.is_recurrent and batch_size > self.n_workers:
            if is_root:
                logging.warning(
                    f"Batchsize is larger than possible with the available number of independent sequences for "
                    f"Truncated BPTT. Setting batchsize to {self.n_workers}, which means {self.n_workers * self.tbptt_length} "
                    f"transitions per batch.")

            batch_size = self.n_workers

        # rebuild model with desired batch size
        joint_weights = self.joint.get_weights()
        self.policy, self.value, self.joint = self.build_models(
            weights=joint_weights,
            batch_size=batch_size // n_optimizers,
            sequence_length=self.tbptt_length
        )

        actor = self._make_actor()

        cycle_start = None
        full_drill_start_time = time.time()
        for self.iteration in range(self.iteration, n):
            mpi_flat_print(f"Gathering cycle {self.iteration}...")

            time_dict = OrderedDict()
            subprocess_start = time.time()

            # distribute parameters from rank 0 to all goal ranks
            joint_weights = mpi_comm.bcast(joint_weights, root=0)
            self.policy, self.value, self.joint = self.build_models(
                weights=joint_weights, batch_size=batch_size // n_optimizers, sequence_length=self.tbptt_length)

            # run simulations in parallel
            worker_stats = []

            _, _, actor_joint = self.build_models(joint_weights, 1, 1)

            for i in worker_collection_ids:
                stats = actor.collect(actor_joint, self.env, self.distribution, self.horizon,
                                      self.discount, self.lam,
                                      self.tbptt_length, collector_id=i)
                worker_stats.append(stats)

            # merge gatherings from all workers
            stats = mpi_condense_stats(worker_stats)
            stats = mpi_comm.bcast(stats, root=0)

            # sync the environments to share statistics for transformers etc.
            self.env.mpi_sync()

            time_dict["gathering"] = time.time() - subprocess_start
            subprocess_start = time.time()

            # make seperate evaluation if necessary and wanted
            stats_with_evaluation = stats
            if separate_eval:
                if radical_evaluation or stats.numb_completed_episodes < MIN_STAT_EPS:
                    mpi_flat_print("Evaluating...")
                    n_evaluations = MIN_STAT_EPS if radical_evaluation else MIN_STAT_EPS - stats.numb_completed_episodes
                    evaluation_stats, _ = self.evaluate(n_evaluations)

                    if radical_evaluation:
                        stats_with_evaluation = evaluation_stats
                    else:
                        stats_with_evaluation = mpi_condense_stats([stats, evaluation_stats])
            elif MPI.COMM_WORLD.rank == 0 and stats.numb_completed_episodes == 0:
                print("WARNING: You are using a horizon that caused this cycle to not finish a single episode. "
                      "Consider activating separate evaluation in drill() to get meaningful statistics.")

            # RECORD STATS, SAVE MODEL AND REPORT
            if is_root:
                self.record_wrapper_stats()
                self.record_stats(stats_with_evaluation)

                time_dict["evaluating"] = time.time() - subprocess_start
                subprocess_start = time.time()

                # save if best
                last_score = self.cycle_reward_history[-1]
                if last_score is not None and last_score == ignore_none(max, self.cycle_reward_history):
                    self.save_agent_state(name="best")

                if cycle_start is not None:
                    self.cycle_timings.append(round(time.time() - cycle_start, 2))

                used_memory = 0
                for pid in psutil.pids():
                    p = psutil.Process(pid)
                    if "python" in p.name():
                        used_memory += p.memory_info()[0]
                self.used_memory.append(round(used_memory / 1e9, 2))
                self.report(total_iterations=n)
                cycle_start = time.time()

            # EARLY STOPPING  TODO check how this goes with MPI
            if self.env.spec.reward_threshold is not None and stop_early:
                if np.all(np.greater_equal(self.cycle_reward_history[-5:], self.env.spec.reward_threshold)):
                    print("\rAll catch a breath, we stop the drill early due to the formidable result!")
                    break

            # OPTIMIZATION PHASE
            if is_optimization_process:
                mpi_flat_print("Optimizing...")
                dataset = read_dataset_from_storage(dtype_actions=tf.float32 if self.continuous_control else tf.int32,
                                                    id_prefix=self.agent_id, worker_ids=optimizer_collection_ids,
                                                    responsive_senses=self.policy.input_names)
                self.optimize(dataset, epochs, batch_size // n_optimizers)

                # del dataset
                # gc.collect()

                # FINALIZE
                if is_root:
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
                    self.current_fps = stats.numb_processed_frames / (
                        sum([v for v in time_dict.values() if v is not None]))
                    self.gathering_fps = (stats.numb_processed_frames // min(self.n_workers, MPI.COMM_WORLD.size)) / (
                        time_dict["gathering"])
                    self.optimization_fps = (stats.numb_processed_frames * epochs) / (time_dict["optimizing"])
                    self.time_dicts.append(time_dict)

                    self.optimization_timings.append(time_dict["optimizing"])
                    self.gathering_timings.append(time_dict["gathering"])

                joint_weights = self.joint.get_weights()

            # if is_optimization_process:
            #     pprint([cf._garbage_collectors for cf in _learn_on_batch._list_all_concrete_functions()])

            # cleanup to counteract memory leaks
            del actor_joint
            del self.joint
            del self.value
            del self.policy
            del stats

            gc.collect()
            tf.compat.v1.reset_default_graph()
            tf.keras.backend.clear_session()

        # after training rebuild the network so that it is available to the agent
        self.policy, self.value, self.joint = self.build_models(
            weights=joint_weights, batch_size=batch_size // n_optimizers, sequence_length=self.tbptt_length
        )

        if is_root:
            print(f"Drill finished after {round(time.time() - full_drill_start_time, 2)}serialization.")

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

    def _make_actor(self) -> Gatherer:
        # create the Gatherer
        return Gatherer(MPI.COMM_WORLD.rank, self.agent_id)

    def optimize(self, dataset: tf.data.Dataset, epochs: int, batch_size: int) -> None:
        """Optimize the agent's policy and value network based on a given dataset.
        
        Since data processing is apparently not possible with tensorflow data sets on a GPU, we will only let the GPU
        handle the training, but keep the rest of the data pipeline on the CPU. I am not currently sure if this is the
        best approach, but it might be the good anyways for large data chunks anyways due to GPU memory limits. It also
        should not make much of a difference since gym runs entirely on CPU anyways, hence for every experience
        gathering we need to transfer all Tensors from CPU to GPU, no matter whether the dataset is stored on GPU or
        not. Even more so this applies with running simulations on the cluster.

        Args:
            dataset (tf.data.Dataset): tensorflow dataset containing serialization, a, p(a), r and A as components per data point
            epochs (int): number of epochs to train on this dataset
            batch_size (int): batch size with which the dataset is sampled

        Returns:
            None
        """

        _learn_on_batch = tf.function(learn_on_batch)

        policy_loss_history, value_loss_history, entropy_history = [], [], []
        for epoch in range(epochs):
            # for each epoch, dataset first should be shuffled to break correlation
            if not self.is_recurrent:
                dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)

            # then divide into batches
            batched_dataset = dataset.batch(batch_size, drop_remainder=True)

            # and iterate over it while accumulating losses
            policy_epoch_losses, value_epoch_losses, entropies = [], [], []
            for b in batched_dataset:
                # use the dataset to optimize the model
                with tf.device(self.device):
                    if not self.is_recurrent:
                        grad, ent, pi_loss, v_loss = _learn_on_batch(
                            batch=b, joint=self.joint, distribution=self.distribution,
                            continuous_control=self.continuous_control, clip_values=self.clip_values,
                            gradient_clipping=self.gradient_clipping, clipping_bound=self.clip, c_value=self.c_value,
                            c_entropy=self.c_entropy, is_recurrent=self.is_recurrent)
                        self.optimizer.apply_gradients(zip(grad, self.joint.trainable_variables))
                    else:
                        # truncated back propagation through time
                        # incoming batch shape: (BATCH_SIZE, N_SUBSEQUENCES, SUBSEQUENCE_LENGTH, *STATE_DIMS)
                        # we split along the N_SUBSEQUENCES dimension to get batches of single subsequences that can be
                        # fed into the model chronologically to adhere to statefulness
                        # if the following line throws a CPU to GPU error this is most likely due to too little memory
                        # on the GPU; lower the number of worker/horizon/... in that case TODO solve by microbatching
                        split_batch = {bk: tf.split(bv, bv.shape[1], axis=1) for bk, bv in b.items()}
                        for i in range(len(split_batch["advantage"])):
                            # extract subsequence and squeeze away the N_SUBSEQUENCES dimension
                            partial_batch = {k: tf.squeeze(v[i], axis=1) for k, v in split_batch.items()}

                            # find and apply the gradients
                            grad, ent, pi_loss, v_loss = _learn_on_batch(
                                batch=partial_batch, joint=self.joint, distribution=self.distribution,
                                continuous_control=self.continuous_control, clip_values=self.clip_values,
                                gradient_clipping=self.gradient_clipping, clipping_bound=self.clip,
                                c_value=self.c_value, c_entropy=self.c_entropy, is_recurrent=self.is_recurrent
                            )
                            self.optimizer.apply_gradients(zip(grad, self.joint.trainable_variables))

                            # make partial RNN state resets
                            reset_mask = detect_finished_episodes(partial_batch["done"])
                            reset_states_masked(self.joint, reset_mask)

                            del grad
                            del partial_batch
                            gc.collect()
                        del split_batch
                        gc.collect()

                # todo makes no sense with the split batches, need to average over them
                entropies.append(ent)
                policy_epoch_losses.append(pi_loss)
                value_epoch_losses.append(v_loss)

                # reset RNN states after each outer batch
                self.joint.reset_states()

            # remember some statistics
            policy_loss_history.append(tf.reduce_mean(policy_epoch_losses).numpy().item())
            value_loss_history.append(tf.reduce_mean(value_epoch_losses).numpy().item())
            entropy_history.append(tf.reduce_mean(entropies).numpy().item())

        optimization_comm.Barrier()

        # store statistics in agent history
        self.policy_loss_history.append(statistics.mean(policy_loss_history))
        self.value_loss_history.append(statistics.mean(value_loss_history))
        self.entropy_history.append(statistics.mean(entropy_history))

        del dataset
        del batched_dataset
        del _learn_on_batch

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

    def evaluate(self, n: int, save: bool = False) -> Tuple[StatBundle, Any]:
        """Evaluate the current state of the policy on the given environment for n episodes. Optionally can render to
        visually inspect the performance.

        Args:
            n (int): integer value indicating the number of episodes that shall be run
            save (bool): whether to save the evaluation to the monitor directory

        Returns:
            StatBundle with evaluation results
        """
        values = mpi_comm.bcast(self.joint.get_weights(), root=0)
        self.joint.set_weights(values)

        evaluation_result = evaluate(self.policy, self.env, self.distribution)
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

        # calculate computation spend on different phases of the iteration and the percentages
        time_string = ""
        time_distribution_string = ""
        if len(self.time_dicts) > 0:
            times = [time for time in self.time_dicts[-1].values() if time is not None]
            time_percentages = [str(round(100 * t / sum(times))) for i, t in enumerate(times)]
            time_string = "[" + "|".join([str(round(t, 1)) for t in times]) + "]"
            time_distribution_string = "[" + "|".join(map(str, time_percentages)) + "]"
        if isinstance(self.lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = self.lr_schedule(self.optimizer.iterations)
        else:
            current_lr = self.lr_schedule

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
        if is_root:
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
                f"times: {time_string} {time_distribution_string}; "
                f"took {self.cycle_timings[-1] if len(self.cycle_timings) > 0 else ''}s [{time_left} left]; "
                f"mem: {self.used_memory[-1]}/{round(psutil.virtual_memory()[0] / 1e9)}GB;\n")

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
        del parameters["optimizer"], parameters["lr_schedule"], parameters["model_builder"]

        parameters["c_entropy"] = parameters["c_entropy"].numpy().item()
        parameters["c_value"] = parameters["c_value"].numpy().item()
        parameters["clip"] = parameters["clip"].numpy().item()
        parameters["distribution"] = self.distribution.__class__.__name__
        parameters["transformers"] = self.env.serialize()

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
            raise FileNotFoundError("The given agent ID'serialization save history is empty.")

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

        env = make_env(parameters["env_name"] if force_env_name is None else force_env_name,
                       reward_config=parameters.get("reward_configuration"),
                       transformers=transformers_from_serializations(parameters["transformers"]))
        model_builder = getattr(models, parameters["builder_function_name"])
        distribution = getattr(policies, parameters["distribution"])(env)

        loaded_agent = PPOAgent(model_builder, environment=env, horizon=parameters["horizon"],
                                workers=parameters["n_workers"], learning_rate=parameters["learning_rate"],
                                discount=parameters["discount"], lam=parameters["lam"], clip=parameters["clip"],
                                c_entropy=parameters["c_entropy"], c_value=parameters["c_value"],
                                gradient_clipping=parameters["gradient_clipping"],
                                clip_values=parameters["clip_values"], tbptt_length=parameters["tbptt_length"],
                                lr_schedule=parameters["lr_schedule_type"], distribution=distribution, _make_dirs=False)

        for p, v in parameters.items():
            if p in ["distribution", "transformers"]:
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

        for file in glob(f"{STORAGE_DIR}/{self.agent_id}_data_[0-9]+\.tfrecord"):
            os.remove(file)

    def build_models(self, weights, batch_size, sequence_length=1):
        """Build the models (policy, value, joint) with the provided weights."""
        policy, value, joint = self.model_builder(
            self.env, self.distribution,
            **({"bs": batch_size} if requires_batch_size(self.model_builder) else {}),
            **({"sequence_length": sequence_length} if requires_sequence_length(self.model_builder) else {}))
        joint.set_weights(weights)

        return policy, value, joint