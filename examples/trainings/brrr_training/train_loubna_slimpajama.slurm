#!/bin/bash
#SBATCH --job-name=1.3b-pile
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/loubna/logs/brrr/%x-%j.out

set -x -e

#source /admin/home/loubna/.bashrc

conda activate /fsx/loubna/miniconda3/envs/brr4


echo "START TIME: $(date)"

BRRR_REPO=/fsx/loubna/code/dev/pr/brrr/
SCRIPT_REPO=$BRRR_REPO/examples/gpt2_mqa

pushd $SCRIPT_REPO

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

CONFIG_FILE=$SCRIPT_REPO/configs/config_loubna_slimpajama_1b.yaml

# Use the fast modeling code
export USE_FAST=1
# Use of this module requires that the environment variable
# CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
# operations, noted in the code, that should be scheduled before
# compute kernels to overlap the communication with the computation,
# which is necessary for a speedup but not for correctness so that
# ordering isn't imposed by the scheduler. Setting
# CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
# in the order they are called.
export CUDA_DEVICE_MAX_CONNECTIONS=1


CMD=" \
    main.py \
    --config-file $CONFIG_FILE
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens
#export LD_PRELOAD=$CUDA_HOME/lib/libnccl.so
#export LD_LIBRARY_PATH=$CUDA_HOME/efa/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# TODO: tune this
# export OMP_NUM_THREADS=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
SLURM_JOB_ID=${SLURM_JOB_ID:-$SLURM_JOBID}
srun $SRUN_ARGS --jobid $SLURM_JOB_ID -u bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"
