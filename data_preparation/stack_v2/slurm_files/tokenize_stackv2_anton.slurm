#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=11G # This is essentially 1.1T / 96
#SBATCH --partition=production-cluster
#SBATCH -o /fsx/loubna/logs/tokenize_stack_v2_anton/tokenize_%x-%j.out
#SBATCH -e /fsx/loubna/logs/tokenize_stack_v2_anton/tokenize_%x-%j.err
#SBATCH --qos=high

set -x -e
source /admin/home/loubna/.bashrc
conda activate /fsx/loubna/miniconda3/envs/megatron

# File Path setup
echo "START TIME: $(date)"


# Training Setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID

json_file=$1
FULL_OUT_PATH=$2

mkdir -p $FULL_OUT_PATH
echo "Tokenizing $json_file"
echo "FULL_OUT_PATH for saving is: $FULL_OUT_PATH"

CMD=" \
    /fsx/loubna/code/Megatron-LM/tools/preprocess_data.py \
    --input $json_file \
    --output-prefix $FULL_OUT_PATH/gpt2-preprocessed \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file /fsx/bigcode/bigcode-training/starcoder2-tokenizer/tokenizer.json \
    --dataset-impl mmap \
    --append-eod \
    --json-keys content \
    --workers 32 \
    --chunk-size 100 \
    --log-interval 1000
    "

export LAUNCHER="python \
    "

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

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"
