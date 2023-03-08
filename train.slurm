#!/bin/bash
#SBATCH --job-name=train-santacoder-1b
#SBATCH --nodes=60
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/loubna/logs/santacoder/%x-%j.out

set -x -e

source /admin/home/loubna/.bashrc


conda activate megatron

echo "START TIME: $(date)"

SCRIPT_REPO=/fsx/loubna/code/Megatron-LM

pushd $SCRIPT_REPO

LOG_PATH=$SCRIPT_REPO/main_log.txt

# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# File path setup
CHECKPOINT_PATH=/fsx/loubna/experiments/santacoder  # Adjust: Directory to store the checkpoints
DATA_PATH=/fsx/loubna/data/santacoder/gpt2-preprocessed_content_document  # Adjust: Prefix of the preprocessed dataset.
TOKENIZER_FILE=/fsx/loubna/data/tokenizer/digit-bytelevel-bpe-jss-v1.1-49152/tokenizer.json # Adjust


GPT_ARGS="\
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --recompute-granularity full \
       --recompute-method uniform \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --attention-head-type multiquery \
       --init-method-std 0.022 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --micro-batch-size 2 \
       --global-batch-size 960 \
       --lr 0.0002 \
       --train-iters 200000 \
       --lr-decay-iters 200000 \
       --lr-decay-style cosine \
       --lr-warmup-iters 175 \
       --weight-decay .1 \
       --adam-beta2 .95 \
       --clip-grad 1.0 \
       --fp16 \
       --log-interval 10 \
       --save-interval 4000 \
       --eval-interval 200 \
       --eval-iters 10 \
"

TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"


CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFileWithFIM \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    $TENSORBOARD_ARGS \
    --wandb-entity-name loubnabnl \
    --wandb-project-name santacoder_cluster \
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


# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"
