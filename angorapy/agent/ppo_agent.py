#!/usr/bin/env python
"""Implementation of Proximal Policy Optimization Algorithm."""
import collections
import gc
import json
import os
import random
import re
import statistics
import time
from collections import OrderedDict
from glob import glob
from json import JSONDecodeError
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union

import gymnasium as gym
import numpy as np
import nvidia_smi
import psutil
import tensorflow as tf
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gymnasium.spaces import MultiDiscrete
from psutil import NoSuchProcess
from tqdm import tqdm

from angorapy import models
from angorapy.agent.dataio import read_dataset_from_storage
from angorapy.agent.gather import evaluate
from angorapy.agent.gather import Gatherer
from angorapy.agent.ppo.train_step import ff_train_step
from angorapy.agent.ppo.train_step import recurrent_train_step
from angorapy.common import const
from angorapy.common import policies
from angorapy.common.const import BASE_SAVE_PATH
from angorapy.common.const import COLORS
from angorapy.common.const import MIN_STAT_EPS
from angorapy.common.const import PATH_TO_EXPERIMENTS
from angorapy.common.const import PRETRAINED_COMPONENTS_PATH
from angorapy.common.const import STORAGE_DIR
from angorapy.common.mpi_optim import MpiAdam
from angorapy.common.policies import BasePolicyDistribution
from angorapy.common.policies import CategoricalPolicyDistribution
from angorapy.common.policies import GaussianPolicyDistribution
from angorapy.common.postprocessors import BaseRunningMeanPostProcessor
from angorapy.common.postprocessors import postprocessors_from_serializations
from angorapy.common.senses import Sensation
from angorapy.tasks.registration import make_task
from angorapy.tasks.wrappers import TaskWrapper
from angorapy.utilities.core import env_extract_dims
from angorapy.utilities.core import find_optimal_tile_shape
from angorapy.utilities.core import flatten
from angorapy.utilities.core import mpi_flat_print
from angorapy.utilities.core import mpi_print
from angorapy.utilities.datatypes import condense_stats
from angorapy.utilities.datatypes import mpi_condense_stats
from angorapy.utilities.datatypes import StatBundle
from angorapy.utilities.error import ComponentError
from angorapy.utilities.model_utils import get_component
from angorapy.utilities.model_utils import get_layer_names
from angorapy.utilities.model_utils import is_recurrent_model
from angorapy.utilities.model_utils import requires_batch_size
from angorapy.utilities.model_utils import requires_sequence_length
from angorapy.utilities.model_utils import validate_env_model_compatibility
from angorapy.utilities.model_utils import validate_model_builder
from angorapy.utilities.statistics import ignore_none

try:
    from mpi4py import MPI
except:
    MPI = None


class PPOAgent:
    """Agent using the Proximal Policy Optimization Algorithm for learning.

    The default is an implementation using two independent models for the critic and the actor. This is of course more
    expensive than using shared parameters because we need two forward and backward calculations
    per batch however this is what is used in the original paper and most implementations. During development this also
    turned out to be beneficial for performance relative to episodes seen in easy tasks (e.g. CartPole) and crucial
    to make any significant progress in more difficult envs such as LunarLander.
    """

    policy: tf.keras.Model
    value: tf.keras.Model
    joint: tf.keras.Model

    def __init__(
            self,
            model_builder: Callable[..., Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]],
            environment: TaskWrapper,
            horizon: int = 1024,
            workers: int = 8,
            learning_rate: float = 0.001,
            discount: float = 0.99,
            lam: float = 0.95,
            clip: float = 0.2,
            c_entropy: float = 0.01,
            c_value: float = 0.5,
            gradient_clipping: float = None,
            clip_values: bool = True,
            tbptt_length: int = 16,
            lr_schedule: str = None,
            distribution: BasePolicyDistribution = None,
            reward_configuration: str = None,
            _make_dirs=True,
            debug: bool = False,
            pretrained_components: list = None,
            n_optimizers: int = None
    ):
        """ Initialize the PPOAgent with given hyperparameters. Policy and value network will be freshly initialized.

        Agent using the Proximal Policy Optimization Algorithm for learning.

        The default is an implementation using two independent models for the critic and the actor. This is of course more
        expensive than using shared parameters because we need two forward and backward calculations
        per batch however this is what is used in the original paper and most implementations. During development this also
        turned out to be beneficial for performance relative to episodes seen in easy tasks (e.g. CartPole) and crucial
        to make any significant progress in more difficult environments such as LunarLander.

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
            _make_dirs (bool): internal parameter to indicate whether to recreate the directories
            debug (bool): turn on/off debugging mode
        """
        super().__init__()
        self.debug = debug

        # setup distributed computation
        self.gpus = tf.config.list_physical_devices('GPU')
        if MPI is not None:
            self.mpi_comm = MPI.COMM_WORLD
            self.is_root = self.mpi_comm.rank == 0
            self.comm_size = self.mpi_comm.size
            self.comm_rank = self.mpi_comm.rank

            self.optimization_comm, self.is_optimization_process = self.get_optimization_comm(
                limit_to_n_optimizers=n_optimizers)
            self.n_optimizers = int(self.mpi_comm.allreduce(self.is_optimization_process, op=MPI.SUM))
        else:
            self.mpi_comm = None
            self.comm_size = 1
            self.comm_rank = 0
            self.is_root = True
            self.optimization_comm = None
            self.is_optimization_process = True
            self.n_optimizers = 1

        # checkups
        assert lr_schedule is None or isinstance(lr_schedule, str)

        # environment info
        self.env: TaskWrapper = environment
        self.env_name = self.env.unwrapped.spec.id
        self.state_dim, self.n_actions = env_extract_dims(self.env)
        if isinstance(self.env.action_space, (Discrete, MultiDiscrete)):
            self.continuous_control = False
        elif isinstance(self.env.action_space, Box):
            self.continuous_control = True
        else:
            raise NotImplementedError(f"PPO cannot handle unknown Action Space Typ: {self.env.action_space}")
        self.env.warmup()

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
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate,
                                                                              decay_steps=workers * horizon,
                                                                              # decay after every cycle
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
        self.policy, self.value, self.joint = model_builder(env=self.env, distribution=self.distribution,
                                                            **({"bs": 1} if requires_batch_size(model_builder) else {}),
                                                            **({
                                                                   "sequence_length": self.tbptt_length} if requires_sequence_length(
                                                                model_builder) else {}))

        validate_model_builder(self.model_builder, self.env, self.distribution)
        validate_env_model_compatibility(self.env, self.joint)

        if pretrained_components is not None:
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

        self.optimizer: MpiAdam = MpiAdam(comm=self.optimization_comm, learning_rate=self.lr_schedule, epsilon=1e-5)
        self.optimizer.apply_gradients(
            zip([tf.zeros_like(v) for v in self.joint.trainable_variables], self.joint.trainable_variables))

        self.is_recurrent = is_recurrent_model(self.policy)
        if not self.is_recurrent:
            self.tbptt_length = 1

        self.gatherer_class = Gatherer

        # miscellaneous
        self.iteration = 0
        self.current_fps = 0
        self.gathering_fps = 0
        self.optimization_fps = 0
        self.device = "CPU:0"
        self.model_export_dir = "storage/saved_models/exports/"
        self.agent_id = f"{round(time.time())}{random.randint(int(1e5), int(1e6) - 1)}"
        if MPI is not None:
            self.agent_id = self.mpi_comm.bcast(self.agent_id, root=0)

        self.agent_directory = f"{BASE_SAVE_PATH}/{self.agent_id}/"
        self.experiment_directory = f"{PATH_TO_EXPERIMENTS}/{self.agent_id}/"
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
        self.years_of_experience = 0
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
        self.loading_history = []
        self.current_per_receptor_mean = {}
        self.auxiliary_performances = {}

        # training statistics
        self.cycle_timings = []
        self.optimization_timings = []
        self.gathering_timings = []
        self.used_memory = []
        self.used_gpu_memory = []

        self.wrapper_stat_history = {}
        for postprocessor in self.env.postprocessors:
            self.wrapper_stat_history.update(
                {postprocessor.__class__.__name__: {"mean": [postprocessor.simplified_mean()],
                                                    "stdev": [postprocessor.simplified_stdev()]}})

    @staticmethod
    def get_optimization_comm(limit_to_n_optimizers: int = None):
        mpi_comm = MPI.COMM_WORLD
        gpus = tf.config.list_physical_devices('GPU')

        if len(gpus) > 0:
            node_names = list(set(mpi_comm.allgather(MPI.Get_processor_name())))
            node_comm = mpi_comm.Split(color=node_names.index(MPI.Get_processor_name()))
            node_optimizer_rank = node_comm.allreduce(mpi_comm.rank, op=MPI.MIN)

            # TODO make this work for multiple GPUs per node?
            is_optimization_process = MPI.COMM_WORLD.rank == node_optimizer_rank
        else:
            is_optimization_process = True

        if not is_optimization_process:
            try:
                if tf.config.get_visible_devices("GPU"):
                    tf.config.set_visible_devices([], "GPU")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            except:
                pass
        else:
            try:
                if len(gpus) > 0:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                pass

        optimization_ranks = [r for r in mpi_comm.allgather(MPI.COMM_WORLD.rank if is_optimization_process else -1) if
                              r != -1]
        if limit_to_n_optimizers is not None and limit_to_n_optimizers != 0:
            optimization_ranks = optimization_ranks[:limit_to_n_optimizers]
            is_optimization_process = mpi_comm.rank in optimization_ranks

        optimization_comm = mpi_comm.Split(color=(1 if is_optimization_process else 0))

        return optimization_comm, is_optimization_process

    def record_wrapper_stats(self) -> None:
        """Records the stats from RunningMeanWrappers."""
        for postprocessor in self.env.postprocessors:
            if postprocessor.name not in self.wrapper_stat_history.keys() or not isinstance(postprocessor,
                                                                                          BaseRunningMeanPostProcessor):
                continue

            self.wrapper_stat_history[postprocessor.__class__.__name__]["mean"].append(postprocessor.simplified_mean())
            self.wrapper_stat_history[postprocessor.__class__.__name__]["stdev"].append(postprocessor.simplified_stdev())

    def __repr__(self):
        return f"PPOAgent[at {self.iteration}][{self.env_name}]"

    def set_gpu(self, activated: bool) -> None:
        """Set GPU usage mode."""
        self.device = "GPU:0" if activated else "CPU:0"
        pass

    def assign_gatherer(self, new_gathering_class: Callable):
        self.gatherer_class = new_gathering_class

    def act(self, state: Union[Sensation, Dict[str, Any]]):
        """Sample an action from the agent's policy based on a given state. The sampled action is returned in a format
        that can be directly given to an environment.

        This method is mostly useful at inference time and serves as q quick wrapper around the steps required to
        process the raw numpy states of the environment into a state readable by the policy network, followed by
        sampling from the predicted distribution."""

        # based on given state, predict action distribution and state value; need flatten due to tf eager bug
        if isinstance(state, (dict, collections.OrderedDict)) and "observation" in state.keys():
            state = state["observation"]

            if not isinstance(state, Sensation):
                try:
                    state = Sensation(**state)
                except:
                    raise ValueError("Observation in state dict must be a Sensation or dict convertible into "
                                     "an observation.")
        elif isinstance(state, Sensation):
            pass
        else:
            raise ValueError("State must be a Sensation or a dictionary with an 'observation' key.")

        _, _, joint = self.build_models(self.joint.get_weights(), 1, 1)

        prepared_state = state.with_leading_dims(time=self.is_recurrent).dict_as_tf()
        policy_out = flatten(joint(prepared_state, training=False))

        predicted_distribution_parameters, value = policy_out[:-1], policy_out[-1]
        # from the action distribution sample an action and remember both the action and its probability
        action, action_probability = self.distribution.act(*predicted_distribution_parameters)

        return action

    def drill(
            self,
            n: int = 100,
            epochs: int = 3,
            batch_size: int = 256,
            monitor: "Monitor" = None,
            save_every: int = 0,
            separate_eval: bool = False,
            stop_early: bool = False,
            radical_evaluation: object = False
    ) -> "PPOAgent":
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
                above the envs threshold


        Returns:
            self, for chaining
        """

        # start monitor
        if self.is_root and monitor is not None:
            monitor.make_metadata(additional_hps={"epochs_per_cycle": epochs, "batch_size": batch_size, })

        self.batch_size = batch_size

        # determine the number of independent repeated gatherings required on this worker
        worker_base, worker_extra = divmod(self.n_workers, self.comm_size)
        worker_split = [worker_base + (r < worker_extra) for r in range(self.comm_size)]
        worker_collection_ids = list(range(self.n_workers))[
                                sum(worker_split[:self.comm_rank]):sum(worker_split[:self.comm_rank + 1])]

        # determine the split of worker outputs over optimizers
        optimizer_base, optimizer_extra = divmod(self.n_workers, self.n_optimizers)
        optimizer_split = [optimizer_base + (r < optimizer_extra) for r in range(self.n_optimizers)]
        optimizer_collection_ids = list(range(self.n_workers))[sum(optimizer_split[:self.optimizer.comm_rank]):sum(
            optimizer_split[:self.optimizer.comm_rank + 1])]

        if MPI is not None:
            list_of_worker_collection_id_lists = self.mpi_comm.gather(worker_collection_ids, root=0)
            list_of_optimizer_collection_id_lists = self.optimizer.comm.gather(optimizer_collection_ids, root=0)
        else:
            list_of_worker_collection_id_lists = [worker_collection_ids]
            list_of_optimizer_collection_id_lists = [optimizer_collection_ids]

        if self.is_root:
            print(f"\n\nDrill started using {self.comm_size} processes for {self.n_workers} workers of which "
                  f"{self.n_optimizers} are optimizers."
                  f" Worker distribution: {[worker_base + (r < worker_extra) for r in range(self.comm_size)]}.\n"
                  f"IDs over Workers: {list_of_worker_collection_id_lists}\n"
                  f"IDs over Optimizers: {list_of_optimizer_collection_id_lists}")

        if self.is_recurrent:
            assert batch_size % self.tbptt_length == 0, f"Batch size (the number of transitions per update)" \
                                                        f" must be a multiple of the sequence length "

            n_chunks_per_trajectory = self.horizon // self.tbptt_length
            n_chunks_per_batch = batch_size // self.tbptt_length
            n_chunks_per_batch_per_process = n_chunks_per_batch // self.n_optimizers
            n_trajectories_per_process = self.n_workers // self.n_optimizers

            n_trajectories_per_batch_per_process, n_chunks_per_trajectory_per_batch_per_process = find_optimal_tile_shape(
                (n_trajectories_per_process, n_chunks_per_trajectory), n_chunks_per_batch_per_process, width_first=True
                # TODO smartly adapt this to memory reqs or at least make parameter
            )

            n_trajectories_per_batch = n_trajectories_per_batch_per_process * self.n_optimizers
            n_chunks_per_trajectory_per_batch = n_chunks_per_trajectory_per_batch_per_process
            self.n_updates_per_epoch = (self.n_workers * self.horizon) // batch_size

            if self.is_root:
                print(f"\nThe policy is recurrent and the batch size is interpreted as the number of transitions "
                      f"per policy update. Given the batch size of {batch_size} this results in: \n"
                      f"\t{n_chunks_per_batch} chunks per update and {self.n_updates_per_epoch} updates per epoch\n"
                      f"\tBatch tilings of "
                      f"{n_trajectories_per_batch_per_process, n_chunks_per_trajectory_per_batch_per_process} per process "
                      f"and {n_trajectories_per_batch, n_chunks_per_trajectory_per_batch} in total.\n\n")

            batch_size = n_trajectories_per_batch_per_process
            effective_batch_size = (n_trajectories_per_batch_per_process, n_chunks_per_trajectory_per_batch_per_process)

        else:
            assert self.horizon * self.n_workers >= batch_size, "Batch Size is larger than the number of transitions."
            batch_size = batch_size // self.n_optimizers
            effective_batch_size = batch_size

        # rebuild model with desired batch size
        joint_weights = self.joint.get_weights()
        self.policy, self.value, self.joint = self.build_models(weights=joint_weights, batch_size=batch_size,
                                                                sequence_length=self.tbptt_length)

        _, _, actor_joint = self.build_models(joint_weights, 1, 1)

        # reset optimizer to not be confused by the newly build models
        # todo maybe its better to maintain a class attribute with optimizer weights and always write it in here
        optimizer_weights = self.optimizer.get_weights()
        if len(optimizer_weights) > 0:
            self.optimizer = MpiAdam(self.optimizer.comm, self.optimizer.learning_rate, self.optimizer.epsilon)
            self.optimizer.apply_gradients(
                zip([tf.zeros_like(v) for v in self.joint.trainable_variables], self.joint.trainable_variables))
            self.optimizer.set_weights(optimizer_weights)

        actor = self._make_actor(horizon=self.horizon, discount=self.discount, lam=self.lam,
                                 subseq_length=self.tbptt_length)

        cycle_start = None
        full_drill_start_time = time.time()
        for self.iteration in range(self.iteration, n):
            mpi_flat_print(f"Gathering cycle {self.iteration}...")

            time_dict = OrderedDict()
            subprocess_start = time.time()

            # distribute parameters from rank 0 to all goal ranks
            joint_weights = self.mpi_comm.bcast(joint_weights, root=0) if MPI is not None else joint_weights
            self.joint.set_weights(joint_weights)
            actor_joint.set_weights(joint_weights)

            # run simulations in parallel
            worker_stats = []

            for i in worker_collection_ids:
                stats = actor.collect(actor_joint, self.env, collector_id=i)
                worker_stats.append(stats)

            # merge gatherings from all workers
            stats = mpi_condense_stats(worker_stats)

            if MPI is not None:
                stats = self.mpi_comm.bcast(stats, root=0)

            # sync the envs to share statistics for postprocessors etc.
            self.env.mpi_sync()

            time_dict["gathering"] = time.time() - subprocess_start
            subprocess_start = time.time()

            # make separate evaluation if necessary and wanted
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
            elif self.comm_rank == 0 and stats.numb_completed_episodes == 0:
                print("WARNING: You are using a horizon that caused this cycle to not finish a single episode. "
                      "Consider activating separate evaluation in drill() to get meaningful statistics.")

            # RECORD STATS, SAVE MODEL AND REPORT
            if self.is_root:
                self.record_wrapper_stats()
                self.record_stats(stats_with_evaluation)

                time_dict["evaluating"] = time.time() - subprocess_start

                # save if best
                last_score = self.cycle_reward_history[-1]
                if last_score is not None and last_score == ignore_none(max, self.cycle_reward_history):
                    self.save_agent_state(name="best")

                if cycle_start is not None:
                    self.cycle_timings.append(round(time.time() - cycle_start, 2))

                used_memory = 0
                for pid in psutil.pids():
                    try:
                        p = psutil.Process(pid)
                        if "python" in p.name():
                            used_memory += p.memory_info()[0]
                    except NoSuchProcess:
                        pass

                used_gpu_memory = 0
                if len(self.gpus) > 0:
                    nvidia_smi.nvmlInit()
                    nvidia_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                    procs = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(nvidia_handle)
                    used_gpu_memory = 0
                    for p in procs:
                        try:
                            if "python" in psutil.Process(p.pid).name():
                                used_gpu_memory += p.usedGpuMemory
                        except NoSuchProcess:
                            pass

                self.used_memory.append(round(used_memory / 1e9, 2))
                self.used_gpu_memory.append(round(used_gpu_memory / 1e9, 2))
                self.report(total_iterations=n)
                cycle_start = time.time()

            # EARLY STOPPING  TODO check how this goes with MPI
            if self.env.spec.reward_threshold is not None and stop_early:
                if np.all(np.greater_equal(self.cycle_reward_history[-5:], self.env.spec.reward_threshold)):
                    print("\rAll catch a breath, we stop the drill early due to the formidable result!")
                    break

            # OPTIMIZATION PHASE
            if self.is_optimization_process:
                dataset = read_dataset_from_storage(dtype_actions=tf.float32 if self.continuous_control else tf.int32,
                                                    id_prefix=self.agent_id, worker_ids=optimizer_collection_ids,
                                                    responsive_senses=self.policy.input_names)

                subprocess_start = time.time()
                self.optimize(dataset, epochs, batch_size, effective_batch_size)
                time_dict["optimizing"] = time.time() - subprocess_start

                # FINALIZE
                if self.is_root:
                    mpi_flat_print("Finalizing...")
                    self.total_frames_seen += stats.numb_processed_frames
                    self.total_episodes_seen += stats.numb_completed_episodes
                    if hasattr(self.env, "dt"):
                        self.years_of_experience += (stats.numb_processed_frames * self.env.dt / 3.154e+7)
                    else:
                        self.years_of_experience = None

                    # update monitor logs
                    if monitor is not None:
                        if monitor.gif_every != 0 and (self.iteration + 1) % monitor.gif_every == 0:
                            print("Creating Episode GIFs for current state of policy...")
                            monitor.create_episode_gif(n=1)

                        if monitor.frequency != 0 and (self.iteration + 1) % monitor.frequency == 0:
                            monitor.update()

                    # save the current state of the agent
                    if save_every != 0 and self.iteration != 0 and (self.iteration + 1) % save_every == 0:
                        # every x iterations
                        self.save_agent_state()

                    # and (overwrite) the latest version
                    self.save_agent_state("last")

                    # calculate processing speed in fps
                    self.current_fps = stats.numb_processed_frames / (
                        sum([v for v in time_dict.values() if v is not None]))
                    self.gathering_fps = (stats.numb_processed_frames // min(self.n_workers, self.comm_size)) / (
                        time_dict["gathering"])
                    self.optimization_fps = (stats.numb_processed_frames * epochs) / (time_dict["optimizing"])
                    self.time_dicts.append(time_dict)

                    self.optimization_timings.append(time_dict["optimizing"])
                    self.gathering_timings.append(time_dict["gathering"])

                joint_weights = self.joint.get_weights()

            gc.collect()
            tf.compat.v1.reset_default_graph()
            tf.keras.backend.clear_session()

        # after training rebuild the network so that it is available to the agent
        self.policy, self.value, self.joint = self.build_models(weights=joint_weights, batch_size=batch_size,
                                                                sequence_length=self.tbptt_length)

        if self.is_root:
            print(f"Drill finished after {round(time.time() - full_drill_start_time, 2)}serialization.")

        return self

    def record_stats(self, stats):
        """Record a given StatsBundle in the history of the agent."""
        try:
            mean_eps_length = statistics.mean(stats.episode_lengths) if len(stats.episode_lengths) > 1 else \
                stats.episode_lengths[0]
            mean_eps_rewards = statistics.mean(stats.episode_rewards) if len(stats.episode_rewards) > 1 else \
                stats.episode_rewards[0]
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
        self.current_per_receptor_mean = {s: arr.tolist() for s, arr in stats.per_receptor_mean.items()}

        for key, value in stats.auxiliary_performances.items():
            if key not in self.auxiliary_performances:
                self.auxiliary_performances[key] = {"mean": [], "std": []}
            self.auxiliary_performances[key]["mean"].append(np.mean(value).item())
            self.auxiliary_performances[key]["std"].append(np.std(value).item())

    def _make_actor(self, horizon, discount, lam, subseq_length) -> Gatherer:
        # create the Gatherer
        return self.gatherer_class(
            worker_id=self.comm_rank,
            exp_id=self.agent_id,
            distribution=self.distribution,
            horizon=horizon, discount=discount, lam=lam, subseq_length=subseq_length)

    def optimize(self, dataset: tf.data.TFRecordDataset, epochs: int, batch_size: int,
                 effective_batch_sizes: Union[int, Tuple[int, int]]) -> None:
        """Optimize the agent's policy and value network based on a given dataset.

        Args:
            dataset (tf.data.Dataset): tensorflow dataset containing serialization, a, p(a), r and A as components per data point
            epochs (int): number of epochs to train on this dataset
            batch_size (Tuple): batch size representing the number of TRANSITIONS per batch (not chunks)
            effective_batch_sizes (Tuple): batch size representing the number of TRAJECTORIES per batch and the
                                            number of CHUNKS per trajectory per batch

        Returns:
            None
        """

        # start optimization
        if self.is_recurrent:
            total_updates = self.n_updates_per_epoch * epochs
        else:
            total_updates = (self.n_workers * self.horizon) // batch_size * epochs
        policy_loss_history, value_loss_history, entropy_history = [], [], []
        with tqdm(total=total_updates, disable=not self.is_root, desc="Optimizing...", leave=False) as pbar:
            for epoch in range(epochs):
                if self.is_recurrent:
                    batched_dataset = dataset.batch(effective_batch_sizes[0], drop_remainder=True)
                    with tf.device(self.device):
                        super_batch_ents = []
                        super_batch_pi_losses = []
                        super_batch_v_losses = []

                        for super_batch in batched_dataset:
                            ent, pi_loss, v_loss = recurrent_train_step(super_batch=super_batch,
                                                                        batch_size=effective_batch_sizes,
                                                                        joint=self.joint,
                                                                        distribution=self.distribution,
                                                                        continuous_control=self.continuous_control,
                                                                        clip_values=self.clip_values,
                                                                        gradient_clipping=self.gradient_clipping,
                                                                        clip=self.clip, c_value=self.c_value,
                                                                        c_entropy=self.c_entropy,
                                                                        is_recurrent=self.is_recurrent,
                                                                        optimizer=self.optimizer,
                                                                        pbar=pbar if self.is_root else None)

                            super_batch_ents.append(ent)
                            super_batch_pi_losses.append(pi_loss)
                            super_batch_v_losses.append(v_loss)

                        policy_loss_history.append(tf.reduce_mean(super_batch_pi_losses).numpy().item())
                        value_loss_history.append(tf.reduce_mean(super_batch_v_losses).numpy().item())
                        entropy_history.append(tf.reduce_mean(super_batch_ents).numpy().item())
                else:
                    dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)

                    # divide into batches
                    batched_dataset = dataset.batch(batch_size, drop_remainder=True)
                    policy_epoch_losses, value_epoch_losses, entropies = [], [], []

                    with tf.device(self.device):
                        for b in batched_dataset:
                            ent, pi_loss, v_loss = ff_train_step(batch=b, joint=self.joint,
                                                                 distribution=self.distribution,
                                                                 continuous_control=self.continuous_control,
                                                                 clip_values=self.clip_values,
                                                                 gradient_clipping=self.gradient_clipping,
                                                                 clip=self.clip, c_value=self.c_value,
                                                                 c_entropy=self.c_entropy,
                                                                 is_recurrent=self.is_recurrent,
                                                                 optimizer=self.optimizer)
                            pbar.update(1)

                        entropies.append(ent)
                        policy_epoch_losses.append(pi_loss)
                        value_epoch_losses.append(v_loss)

                        # reset RNN states after each outer batch
                        self.joint.reset_states()

                    # remember some statistics
                    policy_loss_history.append(tf.reduce_mean(policy_epoch_losses).numpy().item())
                    value_loss_history.append(tf.reduce_mean(value_epoch_losses).numpy().item())
                    entropy_history.append(tf.reduce_mean(entropies).numpy().item())

        if MPI is not None:
            self.optimization_comm.Barrier()

        # store statistics in agent history
        self.policy_loss_history.append(statistics.mean(policy_loss_history))
        self.value_loss_history.append(statistics.mean(value_loss_history))
        self.entropy_history.append(statistics.mean(entropy_history))

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

    def evaluate(self, n: int, save: bool = False, act_confidently=False) -> Tuple[StatBundle, Any]:
        """Evaluate the current state of the policy on the given environment for n episodes.

        Args:
            n (int): integer value indicating the number of episodes that shall be run
            save (bool): whether to save the evaluation to the monitor directory
            act_confidently (bool): Choose actions deterministically instead of sampling them from the distribution
        Returns:
            StatBundle with evaluation results
        """
        policy, value, joint = self.build_models(self.joint.get_weights(), 1, 1)

        stat_bundles = []
        for i in tqdm(range(n), disable=not self.is_root):
            lengths, rewards, classes, auxiliary_performances = evaluate(policy, self.env, self.distribution,
                                                                         act_confidently)
            stat_bundles.append(StatBundle(1, lengths, [rewards], [lengths], tbptt_underflow=0, per_receptor_mean={},
                                           auxiliary_performances=auxiliary_performances))

        stats = condense_stats(stat_bundles)

        if save and self.is_root:
            os.makedirs(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/", exist_ok=True)

            previous_evaluations = {}
            if os.path.isfile(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/evaluations.json"):
                with open(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/evaluations.json", "r") as jf:
                    previous_evaluations = json.load(jf)

            previous_evaluations.update({self.iteration: stats._asdict()})

            with open(f"{const.PATH_TO_EXPERIMENTS}/{self.agent_id}/evaluations.json", "w") as jf:
                json.dump(previous_evaluations, jf)

        return stats, classes

    def report(self, total_iterations, verbose=False):
        """Print a report of the current state of the training."""
        if not self.is_root:
            return

        sc, nc, ec, ac = COLORS["OKGREEN"], COLORS["OKBLUE"], COLORS["ENDC"], COLORS["FAIL"]
        reward_col = ac
        if hasattr(self.env.spec, "reward_threshold") and self.env.spec.reward_threshold is not None and \
                self.cycle_reward_history[0] is not None and self.cycle_reward_history[-1] is not None:
            half_way_there_threshold = (self.cycle_reward_history[0] + 0.5 * (
                    self.env.spec.reward_threshold - self.cycle_reward_history[0]))
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
        pi_loss = "  pi  " if len(self.policy_loss_history) == 0 else f"{round(self.policy_loss_history[-1], 2):6.2f}"
        v_loss = "  v     " if len(self.value_loss_history) == 0 else f"{round(self.value_loss_history[-1], 2):8.2f}"
        ent = "  ent " if len(self.entropy_history) == 0 else f"{round(self.entropy_history[-1], 2):6.2f}"

        # tbptt underflow
        underflow = f"w: {nc}{self.underflow_history[-1]}{ec}; " if self.underflow_history[-1] is not None else ""

        # timings
        time_left = "unknown time"
        if len(self.cycle_timings) > 0:
            time_left = f"{round(ignore_none(statistics.mean, self.cycle_timings) * (total_iterations - self.iteration) / 60, 1)}mins"

        total_gpu_memory = 0
        if len(self.gpus) > 0:
            nvidia_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            nvidia_info = nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_handle)
            total_gpu_memory = nvidia_info.total

        # years of experience
        years_of_experience_report = ""
        if self.years_of_experience is not None:
            years_of_experience_report = f"y.exp: {nc}{round(self.years_of_experience, 3):3.3f}{ec}; "

        report_items = {
            "cycle": f"{sc}{f'Cycle {self.iteration:5d}/{total_iterations}' if self.iteration != 0 else 'Before Training'}{ec}",
            "reward": f"r: {reward_col}{'-' if self.cycle_reward_history[-1] is None else f'{round(self.cycle_reward_history[-1], 2):8.2f}'}{ec}",
            "length": f"len: {nc}{'-' if self.cycle_length_history[-1] is None else f'{round(self.cycle_length_history[-1], 2):8.2f}'}{ec}",
            "n": f"n: {nc}{'-' if self.cycle_stat_n_history[-1] is None else f'{self.cycle_stat_n_history[-1]:3d}'}{ec}",
            "loss": f"loss: [{nc}{pi_loss}{ec}|{nc}{v_loss}{ec}|{nc}{ent}{ec}]",
            "time": f"time: {time_string} {time_distribution_string}", "underflow": underflow,
            "lr": f"lr: {nc}{current_lr:.2e}{ec}",
            "updates": f"upd: {nc}{self.optimizer.iterations.numpy().item() - 1:6d}{ec}",
            "frames": f"f: {nc}{round(self.total_frames_seen / 1e3, 3):8.3f}{ec}k",
            "time_left": f"time left: {nc}{time_left}{ec}", "yoe": years_of_experience_report,
            "eps": f"eps: {nc}{self.total_episodes_seen:5d}{ec}",
            "took": f"took {self.cycle_timings[-1] if len(self.cycle_timings) > 0 else ''}s [{time_left} left]",
            "mem": f"mem: {self.used_memory[-1]}/{round(psutil.virtual_memory()[0] / 1e9)}|{self.used_gpu_memory[-1]}/{round(total_gpu_memory / 1e9, 2)}"}

        included_items = ["cycle", "reward", "length", "n", "loss", "updates", "yoe", "time", "time_left", "took"]
        if verbose:
            included_items = report_items.keys()

        mpi_flat_print("; ".join([report_items[k] for k in included_items]) + "\n")

    def save_agent_state(self, name=None):
        """Save the current state of the agent into the agent directory, identified by the current iteration."""
        if name is None:
            name = str(self.iteration)

        if not os.path.exists(self.agent_directory + f"/{name}/weights"):
            os.makedirs(self.agent_directory + f"/{name}/weights")

        self.joint.save_weights(os.path.join(self.agent_directory, f"{name}/weights"), overwrite=True)
        with open(self.agent_directory + f"/{name}/parameters.json", "w") as f:
            json.dump(self.get_parameters(), f)

        np.savez(self.agent_directory + f"/{name}/optimizer_weights.npz", *self.optimizer.get_weights())

    def get_parameters(self):
        """Get the agents parameters necessary to reconstruct it."""
        parameters = self.__dict__.copy()
        del parameters["env"]
        del parameters["policy"], parameters["value"], parameters["joint"], parameters["distribution"]
        del parameters["optimizer"], parameters["lr_schedule"], parameters["model_builder"], parameters[
            "gatherer_class"]
        del parameters["mpi_comm"], parameters["optimization_comm"], parameters["is_optimization_process"], parameters[
            "n_optimizers"], parameters["gpus"], parameters["is_root"], parameters["comm_rank"], parameters["comm_size"]

        parameters["c_entropy"] = parameters["c_entropy"].numpy().item()
        parameters["c_value"] = parameters["c_value"].numpy().item()
        parameters["clip"] = parameters["clip"].numpy().item()
        parameters["distribution"] = self.distribution.__class__.__name__
        parameters["transformers"] = self.env.serialize()
        parameters["optimizer"] = self.optimizer.serialize()

        return parameters

    @staticmethod
    def from_agent_state(
            agent_id: int,
            from_iteration: Union[int, str] = None,
            force_env_name=None,
            path_modifier="",
            n_optimizers: int = None
    ) -> "PPOAgent":
        """Build an agent from a previously saved state.

        Args:
            agent_id:           the ID of the agent to be loaded
            from_iteration:     from which iteration to load, if None (default) use most recent, can be iteration int
                                or ["b", "best"] for best, if such was saved.
            force_env_name:     if not None, override the environment name saved in the agent state
            path_modifier:      if not None or empty, prepend the given string to the path to the agent
            n_optimizers:       if not None, use a specific number of optimizers for the agent

        Returns:
            loaded_agent: a PPOAgent object of the same state as the one saved into the path specified by agent_id
        """
        if MPI is not None:
            mpi_comm = MPI.COMM_WORLD
            is_root = mpi_comm.rank == 0
            optimization_comm, is_optimization_process = PPOAgent.get_optimization_comm(
                limit_to_n_optimizers=n_optimizers)
        else:
            is_root = True
            optimization_comm, is_optimization_process = None, True

        # TODO also load the state of the optimizers
        agent_path = os.path.join(path_modifier, BASE_SAVE_PATH, f"{agent_id}")

        if not os.path.isdir(agent_path):
            raise FileNotFoundError(
                f"The given agent ID does not match any existing save history from your current path. Searched for "
                f"{os.path.abspath(agent_path)}")

        if len(os.listdir(agent_path)) == 0:
            raise FileNotFoundError("The given agent ID's save history is empty.")

        # determine loading point
        latest_matches = PPOAgent.get_saved_iterations(agent_id, path_modifier=path_modifier)

        if from_iteration is None:
            if len(latest_matches) > 0:
                from_iteration = max(latest_matches)
            else:
                from_iteration = "best"
        elif isinstance(from_iteration, str):
            assert from_iteration.lower() in ["best", "b", "last"], \
                "Unknown string identifier, can only be 'best'/'b'/'last' or int."
            if from_iteration == "b":
                from_iteration = "best"
            if from_iteration == "last" and not os.path.isdir(f"{agent_path}/last"):
                from_iteration = "best"
        else:
            assert from_iteration in latest_matches, "There is no save at this iteration."

        # make stack of fallbacks in case the targeted iteration is corrupted
        fallback_stack = ["best", "last"]
        if len(latest_matches) > 0:
            fallback_stack.append(max(latest_matches))

        can_load = False
        while not can_load:
            try:
                loading_path = f"{agent_path}/{from_iteration}/"
                with open(f"{loading_path}/parameters.json", "r") as f:
                    parameters = json.load(f)

                can_load = True
            except json.decoder.JSONDecodeError as e:
                print(f"The parameter file of {from_iteration} seems to be corrupted. "
                      f"Falling back to {fallback_stack[0]}.")
                print(e)
                from_iteration = fallback_stack.pop(0)

        if is_root:
            print(f"Loading from iteration {from_iteration}.")

        postprocessors = postprocessors_from_serializations(parameters["transformers"])
        env = make_task(parameters["env_name"] if force_env_name is None else force_env_name,
                        reward_config=parameters.get("reward_configuration"),
                        postprocessors=postprocessors,
                        render_mode="rgb_array" if re.match(".*[Vv]is(ion|ual).*", parameters["env_name"]) else None)
        model_builder = models.MODEL_BUILDERS[parameters["builder_function_name"]]
        distribution = getattr(policies, parameters["distribution"])(env)

        loaded_agent = PPOAgent(model_builder, environment=env, horizon=parameters["horizon"],
                                workers=parameters["n_workers"], learning_rate=parameters["learning_rate"],
                                discount=parameters["discount"], lam=parameters["lam"], clip=parameters["clip"],
                                c_entropy=parameters["c_entropy"], c_value=parameters["c_value"],
                                gradient_clipping=parameters["gradient_clipping"],
                                clip_values=parameters["clip_values"], tbptt_length=parameters["tbptt_length"],
                                lr_schedule=parameters["lr_schedule_type"], distribution=distribution, _make_dirs=False,
                                reward_configuration=parameters["reward_configuration"], n_optimizers=n_optimizers)

        for p, v in parameters.items():
            if p in ["distribution", "transformers", "c_entropy", "c_value", "gradient_clipping", "clip", "optimizer"
                     "comm_rank", "comm_size", "is_optimization_process", "n_optimizers", "gpus", "is_root"]:
                continue

            loaded_agent.__dict__[p] = v

        loaded_agent.joint.load_weights(
            f"./{path_modifier}/{BASE_SAVE_PATH}/{agent_id}/" + f"/{from_iteration}/weights")

        if "optimizer" in parameters.keys():  # for backwards compatibility
            if os.path.isfile(agent_path + f"/{from_iteration}/optimizer_weights.npz"):
                optimizer_weights = list(np.load(agent_path + f"/{from_iteration}/optimizer_weights.npz").values())
                parameters["optimizer"]["weights"] = optimizer_weights

                loaded_agent.optimizer = MpiAdam.from_serialization(optimization_comm,
                                                                    parameters["optimizer"],
                                                                    loaded_agent.joint.trainable_variables)
            else:
                loaded_agent.optimizer = MpiAdam.from_serialization(optimization_comm,
                                                                    parameters["optimizer"],
                                                                    loaded_agent.joint.trainable_variables)

            if is_root:
                print("Loaded optimizer weights from file.")

        # mark the loading
        loaded_agent.loading_history.append([loaded_agent.iteration])

        return loaded_agent

    @staticmethod
    def get_saved_iterations(agent_id: int, path_modifier="") -> list:
        """Return a list of iterations at which the agent of given ID has been saved."""
        agent_path = os.path.join(path_modifier, BASE_SAVE_PATH, f"{agent_id}")

        if not os.path.isdir(agent_path):
            raise FileNotFoundError("The given agent ID does not match any existing save history.")

        if len(os.listdir(agent_path)) == 0:
            return []

        its = [int(i.group(0)) for i in [re.match("([0-9]+)", fn) for fn in os.listdir(agent_path)] if i is not None]

        return sorted(its)

    def finalize(self):
        """Perform final steps on the agent that are necessary no matter whether an error occurred or not."""
        for file in glob(fr"{STORAGE_DIR}/{self.agent_id}_data_[0-9]+\.tfrecord"):
            os.remove(file)

    def build_models(self, weights, batch_size, sequence_length=1):
        """Build the models (policy, value, joint) with the provided weights."""
        policy, value, joint = self.model_builder(self.env, self.distribution,
                                                  **({"bs": batch_size} if requires_batch_size(
                                                      self.model_builder) else {}),
                                                  **({"sequence_length": sequence_length} if requires_sequence_length(
                                                      self.model_builder) else {}))
        joint.set_weights(weights)

        return policy, value, joint

    def load_components(self, path_to_components: str):
        for directory in os.listdir(path_to_components):
            pretrained_component_full_path = os.path.join(path_to_components, directory)
            if os.path.isdir(pretrained_component_full_path):
                try:
                    pretrained_component = tf.keras.models.load_model(pretrained_component_full_path, compile=False)
                except JSONDecodeError as e:
                    raise ComponentError("Could not load the pretrained model from given files. Likely, the saved model"
                                         "files where created with a different incompatible TF version.")

                found_counterpart = False
                for model_component in self.joint.layers:
                    if model_component.name == pretrained_component.name:
                        mpi_print(f"Loading weights of '{model_component.name}' from pretrained models.")
                        model_component.set_weights(pretrained_component.get_weights())

                        found_counterpart = True
                        break

                if not found_counterpart:
                    print(f"No counterpart for pretrained component '{pretrained_component.name}' found in the model.")

    def freeze_component(self, component: str):
        for model_component in self.joint.layers:
            if model_component.name == component:
                mpi_print(f"Freezing '{model_component.name}'.")
                model_component.trainable = False
