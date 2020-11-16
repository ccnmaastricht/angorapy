#!/bin/bash
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et

#SBATCH --nodes=2
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=12
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=gpu
#SBATCH --time=08:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

# load modules
module load openmpi/3.0.0
module load daint-gpu
module load cray-python
module load cudatoolkit/10.2.89_3.28-7.0.2.1_2.17__g52c0314

# load virtual environment
source ${HOME}/dexterityvenv/bin/activate

# run it
srun --hint=nomultithread python3 -u train.py LunarLanderContinuous-v2 --config continuous_beta --worker 24