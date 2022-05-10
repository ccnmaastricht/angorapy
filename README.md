<img src="docs/img/logo.png" width=15% align="right" />

# AngoraPy

## Anthropomorphic Goal-Oriented Robotic Control for Neuroscientific Modeling

![](https://img.shields.io/github/license/ccnmaastricht/dexterous-robot-hand)
![](https://img.shields.io/github/issues/ccnmaastricht/dexterous-robot-hand)
![](https://img.shields.io/github/forks/ccnmaastricht/dexterous-robot-hand)
![](https://img.shields.io/github/stars/ccnmaastricht/dexterous-robot-hand)

![Manipulation Gif](docs/gifs/manipulate_best.gif)

## 🧑‍💻 Installation

To install this package, use pip

```bash
pip install angorapy
```

### MuJoCo and MuJoCo-Py
To train on any MuJoCo-based environment, you will need MuJoCo. As of late 2021, MuJoCo is free and can be [downloaded here](https://mujoco.org/download). 
As an interface to python, we use mujoco-py, [available here](https://github.com/openai/mujoco-py). To install both, follow their respective instructions.

If you do not want/can install MuJoCo and/or mujoco-py you can use this framework without MuJoCo. Our implementation automatically checks for a `.mujoco` directory in you home directory. If it does not exist, it will try to avoid loading MuJoCo. However, you can then not load any environments that rely on MuJoCo!

## 🧑‍🎓 Other READMEs
Documentation on specific modules is provided in their respective READMEs.

 - <a href="analysis/README.md">Analysis</a>
 - <a href="monitor/README.md">Monitoring</a>

## 🚀 Getting Started
The scripts `train.py`, `evaluate.py` and `observe.py` provide ready-made scripts for training and evaluating an agent in any environment. With `pretrain.py`, it is possible to pretrain the visual component. `benchmark.py` provides functionality for training a batch of agents possibly using different configs for comparison of strategies.

### Training an Agent
The `train.py` commandline interface provides a convenient entry-point for running all sorts of experiments using the builtin models and environments in angorapy. You can train an agent on any environment with optional hyperparameters. Additionally, a monitor will be automatically linked to the training of the agent. For more detail consult the <a href="monitor/README.md">README on monitoring</a>.

Base usage of `train.py` is as follows:

    python train.py ENV --architecture MODEL
    
For instance, training `LunarLanderContinuous-v2` using the `deeper` architecture is possible by running:

    python train.py LunarLanderContinuous-v2 --architecture deeper
    
For more advanced options like custom hyperparameters, consult

    python train.py -h


### Evaluating an Agent
Use the `evaluate.py` script to easily evaluate a trained agent. Agents are identified by their ID stated in the beginning of a training. You can also find agent IDs in the monitor. Use the script as follows:

    usage: evaluate.py [-h] [-n N] [id]

    Evaluate an agent.

    positional arguments:
      id          id of the agent, defaults to newest

    optional arguments:
      -h, --help  show this help message and exit
      -n N        number of evaluation episodes

## 🔀 Distributed Computation
PPO is an asynchronous algorithm, allowing multiple parallel workers to generate experience independently. 
We allow parallel gathering and optimization through MPI. Agents will automatically distribute their workers evenly on 
the available CPU cores, while optimization is distributed over all available GPUs. If no GPUs are available, all CPUs 
share the task of optimizing.

Distribution is possible locally on your workstation and on HPC sites. 

### 💻 Local Distributed Computing with MPI
To use MPI locally, you need to have a running MPI implementation, e.g. Open MPI 4 on Ubuntu.
To execute `train.py` via MPI, run

```bash
mpirun -np 12 --use-hwthread-cpus python3 train.py ...
```

where, in this example, 12 is the number of locally available CPU threads and `--use-hwthread-cpus`
makes available threads (as opposed to only cores). Usage of `train.py` is as described previously.

### :cloud: Distributed Training on SLURM-based HPC clusters
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

## Citing AngoraPy

If you use AngoraPy for your research, please cite us as follows

    Weidler, T., & Senden, M. (2020). AngoraPy - Anthropomorphic Goal-Oriented Robotic Control for Neuroscientific Modeling [Computer software]

Or using bibtex

    @software{angorapy2020,
        author = {Weidler, Tonio and Senden, Mario},
        month = {3},
        title = {{AngoraPy - Anthropomorphic Goal-Oriented Robotic Control for Neuroscientific Modeling}},
        year = {2020}
    }
