#!/bin/bash -l
#SBATCH --job-name="dexterity"
#SBATCH --account="ich020"
#SBATCH --time=08:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=2
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

# load virtual environment
source ${HOME}/dexterityvenv21/bin/activate

srun python3 -u train.py ReachAbsolute-v0 --pcon hand_beta_no_ent --rcon reach.default --model gru