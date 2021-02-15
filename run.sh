#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --ntasks-per-common=2
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=multithread
#SBATCH --time=08:00:00
#SBATCH --account=ich020

# load modules
module load daint-gpu
module load cray-python
module load cray-mpich

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# load virtual environment
source ${HOME}/dexterityvenv21/bin/activate

# run it
srun  python3 -u train.py FreeReachRelative-v0 --pcon hand_beta --rcon free_reach_positive_reinforcement.default --model gru --workers 24