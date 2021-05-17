#!/bin/bash -l
#SBATCH --job-name="dexterity"
#SBATCH --account="ich020"
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

module load daint-gpu
module load cudatoolkit
module use /apps/daint/UES/6.0.UP04/sandboxes/sarafael/modules/all
module load cuDNN/8.0.3.33

# load virtual environment
source ${HOME}/robovenv/bin/activate

srun python3 -u test_glfw.py