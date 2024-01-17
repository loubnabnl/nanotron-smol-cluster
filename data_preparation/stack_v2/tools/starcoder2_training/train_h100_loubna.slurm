#!/bin/bash
#SBATCH --job-name=7b_32k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:8
#SBATCH --mem-per-cpu=11G # This is essentially 1.1T / 96
#SBATCH --partition=hopper-prod
#SBATCH -o /fsx/loubna/logs/finetune-32k_f/%x-%j-train.out
#SBATCH --qos=high

#sb --array 1-3%1
#sb --dependency=afterany:521974

set -x -e

source ~/.bashrc
export AWS_DEFAULT_REGION=us-east-1


source /admin/home/loubna/miniconda3/etc/profile.d/conda.sh
conda activate /fsx/nouamane/miniconda/envs/2-1-cu121

# activate cluster scpecific variables
module load cuda/12.1

echo "START TIME: $(date)"

# show git commit
echo "Git commit: $(git rev-parse HEAD)"

echo "printenv:"
printenv

echo "nvidia-smi:"
nvidia-smi

echo "torch version:"
python -m torch.utils.collect_env

BRRR_REPO=/fsx/loubna/projects/brrr


GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# get from env or set default
# if CONFIG_FILE is not set
if [ -z ${CONFIG_FILE+x} ]; then
    echo "CONFIG_FILE is unset, using default"
else
    echo "CONFIG_FILE is set to '$CONFIG_FILE'"
    # remove yaml extension
    CONFIG_FILE_NAME=${CONFIG_FILE##*/}
    CONFIG_FILE_NAME=${CONFIG_FILE_NAME%.yaml}

    # copy config file to tmp folder
    TMP_CONFIG_FILE=/fsx/loubna/logs/starcoder2/configs/${CONFIG_FILE_NAME}_${SLURM_JOB_ID}.yaml
    echo "Renaming config file to $TMP_CONFIG_FILE"
    # create TMP_CONFIG_FILE folder if it doesn't exist
    mkdir -p $(dirname $TMP_CONFIG_FILE)
    cp $CONFIG_FILE $TMP_CONFIG_FILE
    CONFIG_FILE=$TMP_CONFIG_FILE
fi


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

export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
unset FI_EFA_ENABLE_SHM_TRANSFER
unset NCCL_PROTO

#    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#    --rdzv_backend c10d \
#    --rdzv-backend etcd-v2 \
#    --rdzv-endpoint etcd.hpc-cluster-hopper.hpc.internal.huggingface.tech:2379 \

CMD=" \
    examples/use_trainer.py \
    --config-file $CONFIG_FILE
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv-id ${SLURM_JOB_ID} \
    --rdzv-backend etcd-v2 \
    --rdzv-endpoint etcd.hpc-cluster-hopper.hpc.internal.huggingface.tech:2379 \
    --tee 3 \
    "

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN  # VERSION, WARN, INFO
export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_ALGO=Ring



# TODO: tune this
# export OMP_NUM_THREADS=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
# SRUN_ARGS=" \
#     --wait=60 \
#     --kill-on-bad-exit=1 \
#     "

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"

# srun doesn't work if this script was called with `srun` itself
# bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"
