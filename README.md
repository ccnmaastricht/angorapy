<img src="docs/img/logo.png" width=15% align="right" />

# Anthropomorphic Robotic Control with Biologically Plausible Networks

![](https://img.shields.io/github/license/ccnmaastricht/dexterous-robot-hand)
![](https://img.shields.io/github/issues/ccnmaastricht/dexterous-robot-hand)
![](https://img.shields.io/github/forks/ccnmaastricht/dexterous-robot-hand)
![](https://img.shields.io/github/stars/ccnmaastricht/dexterous-robot-hand)

![Manipulation Gif](docs/gifs/manipulate_best.gif)

## Installation

### Pip Installation

You can install this repository with pip. Clone it and then from within its root directory run

```bash
pip install -e .
```

#### MuJoCo and MuJoCo-Py
To train on any MuJoCo-based environment, you will need MuJoCo. As of late 2021, MuJoCo is free and can be [downloaded here](https://mujoco.org/download). 
As an interface to python, we use mujoco-py, [available here](https://github.com/openai/mujoco-py). To install both, follow their respective instructions.

If you do not want/can install MuJoCo and/or mujoco-py you can use this framework without MuJoCo. Our implementation automatically checks for a `.mujoco` directory in you home directory. If it does not exist, it will try to avoid loading MuJoCo. However, you can then not load any environments that rely on MuJoCo!

## Other READMEs
Documentation on specific modules is provided in their respective READMEs.

 - <a href="analysis/README.md">Analysis</a>
 - <a href="monitor/README.md">Monitoring</a>

## Usage
The scripts `train.py`, `evaluate.py` and `observe.py` provide ready-made scripts for training and evaluating an agent in any environment. With `pretrain.py`, it is possible to pretrain the visual component. `benchmark.py` provides functionality for training a batch of agents possibly using different configs for comparison of strategies.

### Training an Agent
By using `train.py`, you can train an agent on any environment with optional hyperparameter. Additionally, a monitor will be automatically added to the drilling of the agent, s.t. you can inspect
the training progress. For more detail consult the <a href="monitor/README.md">README on monitoring</a>.

The `train.py` commandline parameters are as follows:

```
usage: train.py [-h] [--architecture {simple,deeper,wider,shadow}] [--model {ffn,rnn,lstm,gru}] [--distribution {categorical,gaussian,beta,multi-categorical}] [--shared]
                [--iterations ITERATIONS] [--pcon PCON] [--rcon RCON] [--cpu] [--sequential] [--load-from LOAD_FROM] [--preload PRELOAD] [--export-file EXPORT_FILE] [--eval]
                [--radical-evaluation] [--save-every SAVE_EVERY] [--monitor-frequency MONITOR_FREQUENCY] [--gif-every GIF_EVERY] [--debug] [--no-monitor] [--workers WORKERS]
                [--horizon HORIZON] [--discount DISCOUNT] [--lam LAM] [--no-state-norming] [--no-reward-norming] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr-pi LR_PI]
                [--lr-schedule {None,exponential}] [--clip CLIP] [--c-entropy C_ENTROPY] [--c-value C_VALUE] [--tbptt TBPTT] [--grad-norm GRAD_NORM] [--clip-values] [--stop-early]
                [env]

Train a PPO Agent on some task.

positional arguments:
  env                   the target gym environment

optional arguments:
  -h, --help            show this help message and exit
  --architecture {simple,deeper,wider,shadow}
                        architecture of the policy
  --model {ffn,rnn,lstm,gru}
                        model type if architecture allows for choices
  --distribution {categorical,gaussian,beta,multi-categorical}
  --shared              make the model share part of the network for policy and value
  --iterations ITERATIONS
                        number of iterations before training ends
  --pcon PCON           config name (utilities/hp_config.py) to be loaded
  --rcon RCON           config (utilities/reward_config.py) of the reward function
  --cpu                 use cpu only
  --sequential          run worker sequentially workers
  --load-from LOAD_FROM
                        load from given agent id
  --preload PRELOAD     load visual component weights from pretraining
  --export-file EXPORT_FILE
                        save policy to be loaded in workers into file
  --eval                evaluate additionally to have at least 5 eps
  --radical-evaluation  only record stats from seperate evaluation
  --save-every SAVE_EVERY
                        save agent every given number of iterations
  --monitor-frequency MONITOR_FREQUENCY
                        update the monitor every n iterations.
  --gif-every GIF_EVERY
                        make a gif every n iterations.
  --debug               run in debug mode (eager mode)
  --no-monitor          dont use a monitor
  --workers WORKERS     the number of workers exploring the environment
  --horizon HORIZON     number of time steps one worker generates per cycle
  --discount DISCOUNT   discount factor for future rewards
  --lam LAM             lambda parameter in the GAE algorithm
  --no-state-norming    do not normalize states
  --no-reward-norming   do not normalize rewards
  --epochs EPOCHS       the number of optimization epochs in each cycle
  --batch-size BATCH_SIZE
                        minibatch size during optimization
  --lr-pi LR_PI         learning rate of the policy
  --lr-schedule {None,exponential}
                        lr schedule type
  --clip CLIP           clipping range around 1 for the objective function
  --c-entropy C_ENTROPY
                        entropy factor in objective function
  --c-value C_VALUE     value factor in objective function
  --tbptt TBPTT         length of subsequences in truncated BPTT
  --grad-norm GRAD_NORM
                        norm for gradient clipping, 0 deactivates
  --clip-values         clip value objective
  --stop-early          stop early if threshold of env was surpassed
```

### Pretraining a Component
The python script `pretrain.py` can be used to train the visual component on one of three bootstrapping tasks: classification, pose estimation and reconstruction. Usage is as follows:

    usage: pretrain.py [-h] [--name NAME] [--load LOAD] [--epochs EPOCHS]
                       [{classify,reconstruct,hands,c,r,h}]

    Pretrain a visual component on classification or reconstruction.

    positional arguments:
      {classify,reconstruct,hands,c,r,h}

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           Name the pretraining to uniquely identify it.
      --load LOAD           load the weights from checkpoint path
      --epochs EPOCHS       number of pretraining epochs

### Evaluating an Agent
Use the `evaluate.py` script to easily evaluate a trained agent. Agents are identified by their ID stated in the beginning of a training. You can also find agent IDs in the monitor. Use the script as follows:

    usage: evaluate.py [-h] [-n N] [id]

    Evaluate an agent.

    positional arguments:
      id          id of the agent, defaults to newest

    optional arguments:
      -h, --help  show this help message and exit
      -n N        number of evaluation episodes

## Distributed Computation
PPO is an asynchronous algorithm, allowing multiple parallel workers to generate experience independently. 
We allow parallel gathering and optimization through MPI. Agents will automatically distribute their workers evenly on 
the available CPU cores, while optimization is distributed over all available GPUs. If no GPUs are available, all CPUs 
share the task of optimizing.

Distribution is possible locally on your workstation and on HPC sites. 

### Local Distributed Computing with MPI
To use MPI locally, you need to have a running MPI implementation, e.g. Open MPI 4 on Ubuntu.
To execute `train.py` via MPI, run

```bash
mpirun -np 12 --use-hwthread-cpus python3 train.py ...
```

where, in this example, 12 is the number of locally available CPU threads and `--use-hwthread-cpus`
makes available threads (as opposed to only cores). Usage of `train.py` is as described previously.

### Distributed Training on SLURM-based HPC clusters
*Please note that the following is optimized and tested on the specific cluster we use, but should extend to at least 
any SLURM based setup.*

On any SLURM-based HPC cluster you may submit your job with sbatch usising the following script template:

```bash
#!/bin/bash -l
#SBATCH --job-name="dexterity"
#SBATCH --account=xxx
#SBATCH --time=24:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu&startx
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

# load virtual environment
source ${HOME}/robovenv/bin/activate

export DISPLAY=:0
srun python3 -u train.py ...
```

The number of parallel workers will equal the number of nodes times the number of CPUs per node 
(32 x 12 = 384 in the template above).


## Customized Usage
We've build this framework with modularity in mind. Many aspects of the training process can be customized through your own implementations. The most common customization is the usage of your own network architecture. The distributed reinforcement learning pipeline requires your model's implementation to follow specific rules. These are specified [here]() alongside a tutorial on how to incorporate your own model into the training process.

### Custom Policy Distributions
Currently, four builtin policy distributions are supported, the Gaussian and the Beta distribution for *continuous* and the (multi-)categorical distribution for *discrete* environments. 

To implement new policy distributions, extend the *BasePolicyDistribution* abstract class.

### Custom Environments

### Custom Reward Functions

### Custom Input and Output Transformers

### Custom Hyperparameters Configurations
Last, and somewhat least, you can add your own preset hyperparameter configuration. 

```python
from dexterity.configs.hp_config import make_config, derive_config

my_conf = make_config(
    batch_size=128,
    workers=8,
    model="lstm",
    architecture="wider"
)

my_sub_conf = derive_config(my_conf,
    {"model": "gru"}
)
```

Hyperparameter configurations created `make_config(...)` automatically assign default values to all required parameters and let you specify only those you want to change. With the help of `derive_config(...)` you can build variants of other configurations, including your own created by either `make_config(...)` or `derive_config(...)`.
