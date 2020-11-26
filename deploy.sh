#!/bin/bash
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et

#SBATCH --nodes=2
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=12
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00

# load modules
module load daint-gpu
module load Horovod
#module load cudatoolkit/10.2.89_3.28-7.0.2.1_2.17__g52c0314

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

# load virtual environment
source ${HOME}/dexterityvenv/bin/activate

# run it
srun python3 -u train.py LunarLanderContinuous-v2 --config continuous_beta --worker 24