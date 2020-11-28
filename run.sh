#!/bin/bash
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et

#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --ntasks-per-core=2
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=multithread
#SBATCH --time=08:00:00

# load modules
module load daint-gpu
module load cray-python/3.8.2.1
module load cray-mpich
module load Horovod/0.19.1-CrayGNU-20.08-tf-2.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

# load virtual environment
source ${HOME}/dexterityvenv/bin/activate

# run it
horovodrun python3 -u train.py LunarLanderContinuous-v2 --config continuous_beta --worker 24