#!/bin/bash

#SBATCH --job-name=dexterity_experiment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=admin@tonioweidler.de
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# this is an adaption of the ray boilerplate https://ray.readthedocs.io/en/latest/deploying-on-slurm.html

worker_num=0 # Must be one less than the total number of nodes

# load modules
module load daint-gpu
module load cray-python
module load cray-nvidia-compute
module load cudatoolkit/10.0.130_3.22-7.0.1.0_5.2__gdfb4ce5
module av graphviz

# load virtual environment
source ${HOME}/dexterityvenv/bin/activate

# get names of allocated nodes and create array
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# choose head node
node1=${nodes_array[0]}

# make head adress
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

# export ip head so pyhton script can read it
export ip_head

# start head node
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password &

# observed sleep time to be required to be high to have head ready before children
sleep 20

for ((  i=1; i<=$worker_num; i++ ))
do
  # start another node
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password &

  sleep 10
done

python3 -u train.py HandTappingAbsolute-v1 --config hand_beta_no_ent --model gru --iterations 10000 --redis-ip $ip_head --redis-pw $redis_password